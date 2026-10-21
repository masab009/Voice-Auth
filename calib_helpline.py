"""Rough in-domain threshold check on real 8 kHz helpline audio (Oct 2026 Urdu calls).

Target trials: two different chunks (pause-split pieces) of the same call -> same caller, SAME session (optimistic:
real verification compares different calls). Non-target trials: chunks of different calls, same gender (a repeat
caller can make two calls, so a few "non-targets" may be the same person -> slightly pessimistic FAR).

    python calib_helpline.py [--calls 400] [--min-dur 4]   -> calib/helpline_trials.json + printed table
"""
import argparse, json, os, random, time

import numpy as np
import pandas as pd

from sv_model import SpeakerModel, decode

# defaults are the helpline machine's paths; --root / --details point it at another call set with the same columns
# (audio relative to root, text, call, chunk, dur, speaker_gender, overlapping_speech, confidence)
ROOT = "/media/vector/Seagate-2TB/urdu-stt/ivr_calls_2026-10/vm_data_mirror"
DETAILS = f"{ROOT}/ivr_calls_2026-10_stt_train_eval_high_and_medium_confidence/urdu/clip_details.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")

ap = argparse.ArgumentParser()
ap.add_argument("--calls", type=int, default=400)
ap.add_argument("--min-dur", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--root", default=ROOT)
ap.add_argument("--details", default=DETAILS)
a = ap.parse_args()
ROOT, DETAILS = a.root, a.details

d = pd.read_csv(DETAILS)
d = d[(d.dur >= a.min_dur) & d.speaker_gender.isin(["male", "female"]) & (d.overlapping_speech.astype(str) == "False")
      & (d.confidence == "high")]
calls = d.groupby("call").filter(lambda g: len(g) >= 2 and g.speaker_gender.nunique() == 1)
ids = sorted(calls.call.unique())
random.Random(a.seed).shuffle(ids)
ids = ids[: a.calls]
rows = []
for c in ids:
    g = calls[calls.call == c].sample(n=2, random_state=a.seed)
    rows += g.to_dict("records")
print(f"{len(ids)} calls, {len(rows)} clips, median {np.median([r['dur'] for r in rows]):.1f} s", flush=True)

m = SpeakerModel()
t = time.time()
E = np.stack([m.embed(decode(os.path.join(ROOT, r["audio"]))) for r in rows])
print(f"embedded in {time.time() - t:.0f}s", flush=True)

S = E @ E.T
calls_arr = np.array([r["call"] for r in rows])
gender = np.array([r["speaker_gender"] for r in rows])
iu = np.triu_indices(len(rows), 1)
same = calls_arr[iu[0]] == calls_arr[iu[1]]
gm = gender[iu[0]] == gender[iu[1]]
tar = S[iu][same]
non = S[iu][~same & gm]


def eer(tar, non):
    th = np.sort(np.concatenate([tar, non]))
    frr = np.searchsorted(np.sort(tar), th, side="left") / len(tar)
    far = 1 - np.searchsorted(np.sort(non), th, side="left") / len(non)
    i = np.argmin(np.abs(frr - far))
    return (frr[i] + far[i]) / 2, th[i]


e, th = eer(tar, non)
ops = {}
for far in (0.01, 0.001, 0.0001):
    thr = float(np.quantile(non, 1 - far))
    ops[f"FAR {far:.2%}"] = {"threshold": round(thr, 4), "FRR": round(float((tar < thr).mean()), 4)}
res = {"targets": int(len(tar)), "nontargets": int(len(non)), "EER": round(float(e), 4), "EER_threshold": round(float(th), 4),
       "target_scores": {"mean": round(float(tar.mean()), 3), "p5": round(float(np.quantile(tar, .05)), 3)},
       "nontarget_scores": {"mean": round(float(non.mean()), 3), "p99": round(float(np.quantile(non, .99)), 3),
                            "max": round(float(non.max()), 3)},
       "operating_points": ops, "calls": len(ids), "min_dur": a.min_dur}
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "helpline_trials.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "helpline_scores.npz"), tar=tar, non=non)
print(json.dumps(res, indent=1))

# the highest-scoring non-target pairs (repeat callers? same recorded prompt?) for a manual look
order = np.argsort(-np.where(~same & gm, S[iu], -9))[:15]
top = [{"score": round(float(S[iu][k]), 3), "a": rows[iu[0][k]]["audio"], "b": rows[iu[1][k]]["audio"],
        "text_a": rows[iu[0][k]]["text"][:90], "text_b": rows[iu[1][k]]["text"][:90]} for k in order]
json.dump(top, open(os.path.join(OUT, "top_nontargets.json"), "w"), ensure_ascii=False, indent=1)
print("non-targets above 0.80:", int((non > 0.80).sum()), "of", len(non), "| above 0.87:", int((non > 0.87).sum()))

# calls are numbered in time order per line; a caller who rings back a few minutes later gets a nearby number.
# Pairs on the same line within NEAR numbers are likely the same person, so leave them out of the non-targets.
NEAR = 100
line = np.array([r["call"].split("/")[0] for r in rows])
num = np.array([int(r["call"].split("_")[-1]) for r in rows])
near = (line[iu[0]] == line[iu[1]]) & (np.abs(num[iu[0]] - num[iu[1]]) <= NEAR)
non2 = S[iu][~same & gm & ~near]
print(f"\nnear-in-time different-call pairs: {int((~same & gm & near).sum())}, of which above 0.80: "
      f"{int((S[iu][~same & gm & near] > 0.80).sum())}; far-apart pairs above 0.80: {int((non2 > 0.80).sum())}")
e2, th2 = eer(tar, non2)
ops2 = {}
for far in (0.01, 0.001, 0.0001):
    thr = float(np.quantile(non2, 1 - far))
    ops2[f"FAR {far:.2%}"] = {"threshold": round(thr, 4), "FRR": round(float((tar < thr).mean()), 4)}
res["excluding_near_calls"] = {"near_window": NEAR, "nontargets": int(len(non2)), "EER": round(float(e2), 4),
                               "EER_threshold": round(float(th2), 4), "nontarget_mean": round(float(non2.mean()), 3),
                               "nontarget_max": round(float(non2.max()), 3), "operating_points": ops2}
json.dump(res, open(os.path.join(OUT, "helpline_trials.json"), "w"), indent=1)
print(json.dumps(res["excluding_near_calls"], indent=1))
