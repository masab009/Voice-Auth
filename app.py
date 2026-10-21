"""Voice authentication demo around sv_model.SpeakerModel (w2v-BERT 2.0 LoRA_Adapter_MFA, LMFT).

    bash run.sh          # http://127.0.0.1:8781

Enrolled speakers live in data/speakers/<name>.json (one embedding per enrollment clip, plus the clips in
data/audio/). Every verify / identify / compare call is appended to data/log.jsonl.
"""
import argparse, datetime, json, os, re, threading, time, uuid

import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from sv_model import SR, SpeakerModel, decode

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SPK = os.path.join(DATA, "speakers")
AUD = os.path.join(DATA, "audio")
LOG = os.path.join(DATA, "log.jsonl")
CFG = os.path.join(HERE, "config.json")
for d in (SPK, AUD):
    os.makedirs(d, exist_ok=True)

# threshold on cosine similarity (0.70: TECHNICAL.md section 5); saving settings on the page writes config.json,
# which overrides these defaults
CONFIG = {"threshold": 0.70, "min_speech_s": 3.0}
if os.path.exists(CFG):
    CONFIG.update(json.load(open(CFG)))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
MODEL = None
LOCK = threading.Lock()  # one GPU forward at a time


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(event, **kw):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": event, **kw}, ensure_ascii=False) + "\n")


def safe_name(name):
    name = (name or "").strip()
    if not re.fullmatch(r"[\w\- .]{1,60}", name):
        abort(400, "name: 1-60 letters, digits, spaces, - _ .")
    return name


def spk_path(name):
    return os.path.join(SPK, name + ".json")


def load_spk(name):
    p = spk_path(name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def template(spk):
    """Enrollment template: mean of the clip embeddings, re-normalised."""
    e = np.mean([c["emb"] for c in spk["clips"]], axis=0)
    return e / (np.linalg.norm(e) + 1e-9)


def speech_seconds(wav):
    """Rough net speech: 30 ms frames within 35 dB of the loudest frame (no VAD model)."""
    n = 480
    f = wav[: len(wav) // n * n].reshape(-1, n)
    if not len(f):
        return 0.0
    db = 10 * np.log10((f ** 2).mean(axis=1) + 1e-10)
    return float((db > db.max() - 35).sum() * n / SR)


def embed_upload(fs):
    """werkzeug FileStorage -> (embedding, info, raw bytes)."""
    raw = fs.read()
    try:
        wav = decode(raw)
    except ValueError as e:
        abort(400, str(e))
    dur = len(wav) / SR
    if dur < 0.5:
        abort(400, "the audio is empty or shorter than 0.5 s")
    t = time.time()
    with LOCK:
        emb = MODEL.embed(wav)
    info = {"duration_s": round(dur, 2), "speech_s": round(speech_seconds(wav), 2),
            "embed_ms": int((time.time() - t) * 1000), "file": fs.filename or "recording"}
    if info["speech_s"] < CONFIG["min_speech_s"]:
        info["warning"] = f"only about {info['speech_s']:.1f} s of speech; aim for {CONFIG['min_speech_s']:.0f} s or more"
    return emb, info, raw


def save_audio(raw, name, kind):
    fn = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{kind}-{uuid.uuid4().hex[:6]}.bin"
    d = os.path.join(AUD, name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, fn), "wb").write(raw)
    return os.path.join(name, fn)


def decision(score):
    return "accept" if score >= CONFIG["threshold"] else "reject"


@app.get("/")
def page():
    return send_file(os.path.join(HERE, "page.html"))


@app.get("/api/config")
def get_config():
    return jsonify(CONFIG)


@app.post("/api/config")
def set_config():
    d = request.get_json(force=True)
    for k in ("threshold", "min_speech_s"):
        if k in d:
            CONFIG[k] = float(d[k])
    json.dump(CONFIG, open(CFG, "w"), indent=1)
    log("config", **CONFIG)
    return jsonify(CONFIG)


@app.get("/api/speakers")
def speakers():
    out = []
    for f in sorted(os.listdir(SPK)):
        if f.endswith(".json"):
            s = json.load(open(os.path.join(SPK, f), encoding="utf-8"))
            out.append({"name": s["name"], "clips": len(s["clips"]),
                        "speech_s": round(sum(c["speech_s"] for c in s["clips"]), 1),
                        "created": s["created"], "updated": s["updated"]})
    return jsonify(out)


@app.post("/api/enroll")
def enroll():
    name = safe_name(request.form.get("name"))
    files = request.files.getlist("audio")
    if not files:
        abort(400, "no audio")
    spk = load_spk(name) or {"name": name, "created": now(), "clips": []}
    added = []
    for fs in files:
        emb, info, raw = embed_upload(fs)
        info["audio"] = save_audio(raw, name, "enroll")
        spk["clips"].append({**info, "time": now(), "emb": emb.round(6).tolist()})
        added.append(info)
    spk["updated"] = now()
    json.dump(spk, open(spk_path(name), "w", encoding="utf-8"), ensure_ascii=False)
    # how consistent the enrollment clips are with each other (low = noisy clip or a different voice)
    embs = [np.array(c["emb"]) for c in spk["clips"]]
    pair = [float(embs[i] @ embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
    log("enroll", name=name, added=[{k: v for k, v in a.items()} for a in added], clips=len(spk["clips"]))
    return jsonify({"name": name, "added": added, "clips": len(spk["clips"]),
                    "consistency": round(min(pair), 3) if pair else None})


@app.delete("/api/speakers/<name>")
def delete_speaker(name):
    name = safe_name(name)
    p = spk_path(name)
    if not os.path.exists(p):
        abort(404)
    os.rename(p, p + f".deleted-{datetime.datetime.now():%Y%m%d-%H%M%S}")
    log("delete", name=name)
    return jsonify({"deleted": name})


@app.post("/api/verify")
def verify():
    name = safe_name(request.form.get("name"))
    spk = load_spk(name)
    if not spk:
        abort(404, f"{name} is not enrolled")
    emb, info, raw = embed_upload(request.files["audio"])
    s = float(template(spk) @ emb)
    per_clip = [round(float(np.array(c["emb"]) @ emb), 3) for c in spk["clips"]]
    res = {"name": name, "score": round(s, 4), "threshold": CONFIG["threshold"], "decision": decision(s),
           "per_clip": per_clip, **info}
    res["audio"] = save_audio(raw, "_tests", "verify")
    log("verify", **res)
    return jsonify(res)


@app.post("/api/identify")
def identify():
    emb, info, raw = embed_upload(request.files["audio"])
    ranks = []
    for f in os.listdir(SPK):
        if f.endswith(".json"):
            spk = json.load(open(os.path.join(SPK, f), encoding="utf-8"))
            ranks.append({"name": spk["name"], "score": round(float(template(spk) @ emb), 4)})
    ranks.sort(key=lambda r: -r["score"])
    best = ranks[0] if ranks else None
    res = {"ranking": ranks, "threshold": CONFIG["threshold"],
           "match": best["name"] if best and best["score"] >= CONFIG["threshold"] else None, **info}
    res["audio"] = save_audio(raw, "_tests", "identify")
    log("identify", **res)
    return jsonify(res)


@app.post("/api/compare")
def compare():
    a, ia, ra = embed_upload(request.files["a"])
    b, ib, rb = embed_upload(request.files["b"])
    s = float(a @ b)
    res = {"score": round(s, 4), "threshold": CONFIG["threshold"], "decision": decision(s),
           "same_speaker": s >= CONFIG["threshold"], "a": ia, "b": ib}
    log("compare", **res)
    return jsonify(res)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def err(e):
    return jsonify(error=getattr(e, "description", str(e))), e.code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8781)
    a = ap.parse_args()
    t = time.time()
    MODEL = SpeakerModel()
    print(f"model loaded in {time.time() - t:.0f}s; threshold {CONFIG['threshold']}", flush=True)
    log("app_started", port=a.port, **CONFIG)
    app.run(host=a.host, port=a.port, threaded=True)
