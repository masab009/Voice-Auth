# Voice-Auth

A local voice authentication (speaker verification and identification) demo built on
**w2v-BERT 2.0 LoRA-Adapter-MFA**, one of the most accurate openly released speaker-embedding models: 0.14 % EER
on VoxCeleb1-O. A small web page lets you enroll people by recording or uploading their voice, then verify a claimed
identity, identify an unknown speaker, or compare two clips.

The demo was set up for an Urdu/Punjabi agricultural helpline. Its default threshold comes from a check on real
8 kHz helpline calls; see [TECHNICAL.md](TECHNICAL.md) for how the system works and how the threshold was chosen.

> **Licence warning.** The model weights and the authors' code are **CC BY-NC-SA 4.0 (non-commercial)**. The model
> was also trained on VoxBlink2, whose licence is non-commercial too. This demo is for evaluation only. Neither the
> weights nor the authors' code are stored in this repository; `download_models.sh` fetches them.

## What it does

| Page tab | What happens |
|---|---|
| **Enroll** | Record or upload 3+ clips of a person. Each clip becomes a 256-number voice embedding, and the person's template is the average. |
| **Verify** | Pick a claimed person and record a test clip. The page shows the score (cosine similarity to the template) against the threshold, and accepts or rejects. |
| **Identify** | Scores one clip against everyone enrolled and ranks them. It names the best match only if that match clears the threshold. |
| **Compare two clips** | Same speaker or not, for any two recordings, with no enrollment. |
| **Speakers & settings** | Lists and deletes enrolled speakers, and changes the threshold. |

Any audio that ffmpeg reads works: wav, mp3, m4a, browser webm, and 8 kHz phone recordings. On an RTX 3090 one
clip takes about 0.1–0.5 s, and the model uses about 2.2 GB of GPU memory.

## Requirements

- Linux with an NVIDIA GPU. About 3 GB of free VRAM is enough; it was developed on an RTX 3090.
- Python 3.12, `ffmpeg`, and `git`. `cloudflared` is optional, for a public https link.
- About 2.5 GB of disk for the checkpoint.

## Install

```bash
git clone https://github.com/masab009/Voice-Auth.git && cd Voice-Auth

python3.12 -m venv .venv
.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA build
.venv/bin/pip install -r requirements.txt

bash download_models.sh
```

`download_models.sh` does two things, and it is safe to re-run:

- **Authors' code:** clones it into `repo/`, pinned to a fixed commit. That code holds the model classes and a
  modified copy of `transformers`.
- **Checkpoint:** downloads `model_lmft_0.14.pth` (2.35 GB) from Hugging Face `zl389/w2v-bert-2.0_SV` into
  `models/`. It resumes a broken download, then checks the file's size and SHA-256.

To use another Python environment instead of `.venv`, set `PYTHON=/path/to/python` when you call `run.sh`.

## Run

```bash
bash run.sh          # loads the model (~10 s) and serves http://127.0.0.1:8781
bash tunnel.sh       # optional: prints a public https://….trycloudflare.com link
```

- **From this machine:** open `http://127.0.0.1:8781`.
- **From another computer or phone:** use the tunnel link. Browsers only allow the microphone on https or localhost.
- **Anyone with the tunnel link can use the page**, and its URL changes every time the tunnel starts.
- **Stopping it:**
  1. Run `ss -ltnp | grep 8781`.
  2. Kill the pid it shows.
  3. If a tunnel is running, kill the `cloudflared` process too.

## Using it well

- **Enroll with 3 or more clips of at least 5 s of speech each**, ideally recorded at different times. After
  enrolling, the page shows the lowest similarity between two enrollment clips. If that is below the threshold, one
  clip is noisy or is someone else, so delete the speaker and enroll again.
- **Test clips need at least 3 s of speech.** 5–10 s is better. Accuracy drops quickly on short clips.
- **Enroll on the channel you will verify on.** A laptop-mic enrollment against a phone-call test scores lower than
  two clips from the same channel.
- **Re-check the threshold for your audio.** The default, **0.70**, was set on 8 kHz phone calls, where the channel
  makes every voice score higher. On clean microphone audio, scores between different people are lower, so you can
  lower the threshold a little, but measure first. `calib_helpline.py` shows how.

## Use from Python

```python
from sv_model import SpeakerModel          # run from this folder, with the venv's python
m = SpeakerModel()                          # loads on the GPU
a = m.embed_file("call_1.wav")              # L2-normalised 256-d numpy vector
b = m.embed_file("call_2.wav")
print(m.score(a, b))                        # cosine similarity; >= 0.70 -> same speaker at the default threshold
```

## HTTP API

All POST endpoints take `multipart/form-data` and return JSON.

| Endpoint | Fields | Returns |
|---|---|---|
| `POST /api/enroll` | `name`, one or more `audio` files | the clips added (duration, estimated speech seconds), the total clip count, and enrollment consistency |
| `POST /api/verify` | `name`, `audio` | `score`, `threshold`, `decision` (accept/reject), and the score against each enrollment clip |
| `POST /api/identify` | `audio` | `ranking` of every enrolled speaker by score, and `match` (or null) |
| `POST /api/compare` | `a`, `b` | `score`, `decision`, `same_speaker` |
| `GET /api/speakers` | | enrolled speakers with clip counts and speech seconds |
| `DELETE /api/speakers/<name>` | | removes a speaker (the file is renamed, not erased) |
| `GET/POST /api/config` | JSON `{"threshold": 0.7, "min_speech_s": 3}` | the current settings |

```bash
curl -F name=ali -F audio=@a1.wav -F audio=@a2.wav -F audio=@a3.wav http://127.0.0.1:8781/api/enroll
curl -F name=ali -F audio=@test.wav http://127.0.0.1:8781/api/verify
```

## Files

| Path | What it is |
|---|---|
| `sv_model.py` | Loads the model; `embed()`, `embed_file()`, `score()`. |
| `app.py`, `page.html` | The Flask app and its single page. |
| `run.sh`, `tunnel.sh` | Start the app; start a public link. |
| `download_models.sh` | Fetches the authors' code and the checkpoint. |
| `config.json` (created when you save settings) | Overrides the defaults in `app.py`: threshold 0.70, minimum speech 3 s. Not committed. |
| `calib_helpline.py`, `calib/helpline_trials.json` | The threshold check on helpline calls and its results. The results are summary numbers only. |
| `TECHNICAL.md` | How the system works, end to end. |
| `data/` (not committed) | Enrolled speakers (`speakers/<name>.json`), their clips (`audio/`), and the event log (`log.jsonl`). |

Enrolled voices and test clips are personal biometric data. `.gitignore` keeps `data/` out of git, so don't
commit or share it.

## Credits

The model and training code are by Ze Li, Ming Cheng and Ming Li: *Enhancing Speaker Verification with W2V-BERT
2.0 and Knowledge Distillation guided Structured Pruning*, arXiv:2510.04213 (2025),
<https://github.com/ZXHY-82/w2v-BERT-2.0_SV>, licensed CC BY-NC-SA 4.0. The base encoder is Meta's w2v-BERT 2.0
(`facebook/w2v-bert-2.0`, MIT).
