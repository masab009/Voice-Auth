# How Voice-Auth works

This document follows a clip from the microphone to an accept/reject decision. It covers the model, the
enrollment and scoring logic, how the threshold was chosen, and what the system does not do yet. File and function
names refer to this repository; `repo/` is the authors' code that `download_models.sh` fetches.

## 1. Overview

```
 audio (any format, any sample rate)
   │  ffmpeg: mono, 16 kHz, float32                         sv_model.decode()
   ▼
 waveform (first 60 s)
   │  80-bin log-mel filterbank, 2-frame stacking           SeamlessM4TFeatureExtractor
   ▼
 160-dim features, one every 20 ms
   │  w2v-BERT 2.0 encoder: projection + 24 Conformer layers   (580 M parameters)
   ▼
 25 hidden-state sequences (1024-dim each)
   │  one small adapter per layer (1024 → 128), concatenated   "Layer Adapter MFA"
   ▼
 3200-dim frames
   │  attentive statistics pooling (weighted mean + std)        ASP
   ▼
 6400-dim utterance vector
   │  linear bottleneck → L2 normalisation
   ▼
 256-dim speaker embedding  ──►  cosine similarity with the enrolled template  ──►  score ≥ threshold ?  accept : reject
```

The whole model is one forward pass. There is no PLDA, no score normalisation and no separate calibration model.
The only tuned quantity in the deployed system is the cosine threshold.

## 2. The model

### 2.1 Encoder: w2v-BERT 2.0

`facebook/w2v-bert-2.0` is Meta's self-supervised speech encoder, the one used in Seamless. It was pre-trained on
about 4.5 M hours of unlabelled audio in 143+ languages, which matters for Urdu, Punjabi and Saraiki speakers.

- **Input:** 80-bin log-mel filterbanks (25 ms windows every 10 ms, normalised per utterance). The feature
  extractor stacks every 2 frames into one 160-dim vector, so the encoder sees 50 frames per second.
- **Structure:**
  - a feature projection (160 → 1024);
  - 24 Conformer layers: hidden size 1024, 16 attention heads, feed-forward 4096, depthwise convolution kernel 31,
    relative-position self-attention.
- **Size:** about 580 M parameters.

The authors load it through their copy of Hugging Face `transformers` (`repo/deeplab/pretrained/audio2vector/module/
transformers`). The copy adds support for structured pruning. We build the encoder from their
`config_prune_tea.json`, the full unpruned "teacher" configuration, and fill every weight from the checkpoint. The
original `facebook/w2v-bert-2.0` weights are therefore never downloaded.

### 2.2 Head: Layer-Adapter multi-layer feature aggregation (MFA)

Speaker identity is spread across depths of the encoder. Lower layers carry more voice timbre and channel
information, upper layers more linguistic content. So the head uses all layers, not just the last one
(`repo/recipes/DeepASV/local/spk_model.py`, class `Audio2Vec_based_Adapter`):

1. **Collect the layers.** It takes all 25 hidden-state sequences: the projected input plus each Conformer layer's
   output (`n_mfa_layers = -1`).
2. **Adapt each layer.** Every layer gets its own small adapter: `Linear(1024→128) → LayerNorm → ReLU →
   Linear(128→128)`. This shrinks each layer to 128 dims and lets the model choose what to keep from each depth.
3. **Concatenate.** The 25 adapter outputs are joined frame by frame into 3,200 dims per frame.
4. **Pool.** **Attentive statistics pooling (ASP)** turns the frames into one vector. A small attention network
   (1×1 convolutions 3200 → 128 → 3200, softmax over time) gives each frame and channel a weight, and the pooled
   vector is the weighted mean concatenated with the weighted standard deviation (6,400 dims). Silent or
   uninformative frames get little weight, which is why the system works without a separate VAD.
5. **Project.** A linear bottleneck maps 6,400 → **256 dims**. That is the speaker embedding.

The head adds about 6.2 M parameters: about 3.7 M in the adapters, 0.8 M in the pooling and 1.6 M in the
bottleneck.

### 2.3 How it was trained (by the authors)

Training used the released configs `models/w2v-bert-2.0_SV/config/v1/s1–s3.yaml`. Every stage trains the model as
a speaker classifier with an **ArcFace (additive angular margin) loss** (scale 32). This pushes embeddings of the
same speaker together on the unit sphere and keeps different speakers at least an angular margin apart. That is why
cosine similarity is the right score at test time.

| Stage | Data | Settings | Effect |
|---|---|---|---|
| 1. Adapter + LoRA, frozen encoder | VoxCeleb2-dev + VoxBlink2: **117,278 speakers** | Encoder frozen except LoRA (rank 64, alpha 128 on attention q and v); 2–3 s crops; MUSAN noise and RIR reverberation augmentation; margin 0.2; 25 epochs; lr 1e-4 | Trains the head and a light encoder update cheaply |
| 2. Joint fine-tuning | Same | LoRA merged into the encoder (`utils/lora_merge.py`), whole model unfrozen; lr 1e-5; 4 epochs | Adapts the whole encoder to speaker discrimination |
| 3. **Large-margin fine-tuning (LMFT)** | VoxCeleb2-dev: 5,994 speakers | Longer 5–6 s crops; no augmentation; **margin 0.5**; lr 1e-5; 2 epochs | Sharpens decisions on long, clean utterances |

The released checkpoint is the stage-3 model (`model_lmft_0.14.pth`). The authors report its error on VoxCeleb1:

| Version | Vox1-O EER | Vox1-E EER | Vox1-H EER |
|---|---:|---:|---:|
| Before LMFT | 0.23 % | 0.38 % | 0.81 % |
| **After LMFT (this checkpoint)** | **0.14 %** | **0.31 %** | **0.73 %** |
| After LMFT + calibration (paper only, not released) | 0.12 % | 0.27 % | 0.55 % |

"Calibration" in the paper is a score back-end fitted on VoxCeleb trials. It was not released, and it would not
transfer to phone audio anyway; section 5 covers our replacement.

### 2.4 Loading and inference (`sv_model.py`)

- **Building the model.** `SpeakerModel()` builds `Audio2Vec_based_Adapter` with exactly the stage-3 arguments
  (`embd_dim 256, adapter_dim 128, ASP, n_mfa_layers -1, config_prune_tea.json`). It then loads
  `checkpoint["modules"]["spk_model"]`.
- **Discarded weights.** The ArcFace classifier in the checkpoint is training-only and is thrown away.
- **Load check.** Loading fails loudly if any weight is missing. The only ones allowed to be absent are the pruning
  "hard-concrete" masks, which the unpruned model does not use. In practice all 933 tensors load, 586.7 M
  parameters in total.
- **Decoding audio.** `decode()` pipes the bytes or file through `ffmpeg -ac 1 -ar 16000 -f f32le`. That converts any
  container, codec or sample rate, so an 8 kHz phone call is upsampled to 16 kHz. Upsampling does not add the
  missing 4–8 kHz band; section 5 covers what that does to scores.
- **Embedding.** `embed()` keeps the first 60 s, as the authors' `max_valid_dur` does, and runs the model under
  `torch.autocast(bfloat16)` as the authors scored. It then L2-normalises the 256-dim output.
- **Speed.** About 2.2 GB of VRAM on an RTX 3090, and 0.1–0.5 s per clip depending on length.
- **One stub.** The authors' `ecapa_tdnn.py` imports `calflops`, a FLOP counter that inference never uses, so
  `sv_model.py` stubs that import out rather than installing it.

## 3. Enrollment, verification, identification (`app.py`)

### 3.1 Enrollment

`POST /api/enroll` with a name and one or more clips does the following:

1. Each clip is decoded and embedded. The clip's duration and a rough **net-speech estimate** are recorded. The
   estimate counts 30 ms frames whose energy is within 35 dB of the loudest frame (`speech_seconds()`).
2. A warning is returned if a clip has under `min_speech_s` (3 s) of speech.
3. The embedding is appended to `data/speakers/<name>.json`, and the raw upload is kept in `data/audio/<name>/`.
   Enrolling again under the same name adds clips.
4. **Template.** The template is the mean of the clip embeddings, re-normalised to unit length (`template()`).
   Averaging several clips cancels out per-clip noise and channel effects. Published results show EER falling
   steeply from 1 to 3 to 10 enrollment segments, which is why the page asks for 3 or more.
5. **Consistency check.** The response includes the **lowest pairwise cosine** between the person's enrollment
   clips. If it is below the threshold, one clip is probably noisy, too short, or a different person.

### 3.2 Verification (1:1)

`POST /api/verify` with a claimed name and a test clip:

```
score    = cos(template(name), embed(test))        # both unit vectors, so this is a dot product, range −1…1
decision = "accept" if score ≥ threshold else "reject"
```

The response also gives the cosine against each enrollment clip, for diagnosis. A test clip that matches one
enrollment clip well but the others poorly points to a channel or session mismatch.

### 3.3 Identification (1:N, open set)

`POST /api/identify` scores the test clip against every template and sorts the results.

- The top speaker is reported as the match **only if their score clears the same threshold**. Otherwise the answer
  is "no enrolled speaker", so a stranger is not forced onto the closest person.
- With N enrolled people, the chance that some impostor clears the threshold grows roughly N times. Large
  galleries need a higher threshold, or a margin rule (top-1 minus top-2).

### 3.4 Compare

`POST /api/compare` is a plain `cos(embed(a), embed(b))` against the threshold, with no stored state.

### 3.5 State and logging

- **Speakers.** `data/speakers/<name>.json` holds the name, timestamps, and per clip the embedding (256 floats),
  duration, speech seconds and the stored audio path. Deleting a speaker renames the file to
  `.deleted-<timestamp>` so it can be audited.
- **Log.** Every enroll, verify, identify, compare and threshold change is appended to `data/log.jsonl`, with the
  scores and decisions.
- **Concurrency.** One GPU forward pass runs at a time (a lock); Flask serves requests on threads.
- **The page.** `page.html` is a single static page. It records with `MediaRecorder` (webm/opus), with browser echo
  cancellation and noise suppression turned off so the voice isn't altered, or it takes file uploads, and it calls
  the API.

## 4. Why cosine on these embeddings works

ArcFace training puts every speaker's embeddings in a tight cone around a class centre on the 256-dim unit sphere,
with an angular margin between cones. Two clips of the same person therefore have a small angle (high cosine), and
different people a large one. The LMFT stage uses a larger margin (0.5) on longer crops. That makes the cones
tighter for utterances of several seconds, the regime a verification call is in.

## 5. Choosing the threshold: a check on real helpline calls

Thresholds tuned on VoxCeleb do not carry over to telephone audio. On 8 kHz calls everything above 4 kHz is
missing, and every caller shares the same codec, network and handset characteristics. That shared channel pushes
all embeddings towards one another, so even different people score well above zero. We measured this directly with
`calib_helpline.py`; the results are in `calib/helpline_trials.json`.

### Trials

**Data:** 400 Urdu calls from the October 2026 helpline export (8 kHz). From each call, two pause-separated chunks
were taken: at least 4 s each, single speaker, no overlapping speech, transcribed with high confidence. The median
chunk is 22.8 s.

**Same-speaker pairs:** the two chunks of one call, 400 pairs. Both chunks come from the *same session*, so these
are easier than a real login, which compares against a different call. The false-reject rates below are therefore
optimistic.

**Different-speaker pairs:** chunks from different calls with the same speaker gender, 249,132 pairs.

**Repeat callers:** callers often ring back minutes later, and those calls get nearby numbers. Among calls within
100 numbers of each other on the same line, 385 of 1,272 pairs scored above 0.80. Reading their transcripts
confirmed many are the same farmer continuing a question ("Bashir bhai, my second question…"). Those pairs are left
out of the impostor set. A few repeat callers on different days probably remain, which makes the false-accept
numbers slightly pessimistic.

### Results

| Statistic | Value |
|---|---:|
| Same-speaker score, mean (5th percentile) | 0.865 (0.736) |
| Different-speaker score, mean | **0.391** (on clean wideband audio this would be near 0) |
| Equal error rate | **1.5 %**, at a threshold of 0.630 |

| Threshold | Wrong person accepted (FAR) | Right person rejected (FRR, same-call) |
|---:|---:|---:|
| 0.644 | 1 % | 1.8 % |
| **0.70 (default)** | between 1 % and 0.1 % | between 1.8 % and 5.8 % |
| 0.737 | 0.1 % | 5.8 % |
| 0.886 | 0.01 % | 51 % |

### What the numbers mean

- **The default is 0.70**, between the 1 % and 0.1 % false-accept points.
- **Phone audio needs a high threshold.** Because different callers average 0.39 on phone audio, a VoxCeleb-style
  threshold of about 0.3–0.4 would accept roughly half of all impostors.
- **0.01 % false accepts is out of reach** with this setup. It would need a cohort-based back-end (below), longer
  speech, or fine-tuning on helpline voices.
- **Clean microphone audio behaves differently.** With both enrollment and test on a laptop or phone mic, scores
  between different people are lower. The same threshold then rejects impostors even more safely, at some cost in
  false rejects. Re-run the check on audio from the channel you will deploy on.

### Reproducing it

```bash
.venv/bin/python calib_helpline.py --calls 400 --min-dur 4 \
    --root /path/to/call/audio --details /path/to/clip_details.csv
```

The `clip_details.csv` needs these columns: `audio`, `text`, `call`, `chunk`, `dur`, `speaker_gender`,
`overlapping_speech`, `confidence`.

## 6. Limits and next steps

| Gap | Why it matters | Fix |
|---|---|---|
| **No anti-spoofing** | A recording or AI clone of an enrolled voice will pass. Replay over a phone line is cheap. | Add an XLS-R-based spoofing detector (e.g. SSL-AASIST, XLSR-SLS) before verification; use random spoken prompts (digits) checked by ASR; add attempt limits. |
| **No score normalisation or calibration** | Raw cosine shifts with channel and language. That is why the threshold must be tuned per channel. | AS-norm against a cohort of helpline callers' embeddings, then logistic-regression calibration on in-domain trials from *different calls*. |
| **Same-session threshold check** | Real logins compare different calls, days and handsets, which gives more false rejects than measured. | Build a cross-call trial list from repeat callers with verified identity. |
| **Energy-based speech estimate, no diarization** | A clip with two voices (caller and agent) mixes their embeddings. | A neural VAD (e.g. Silero), and separate channels for caller and agent in the IVR. |
| **Model not adapted to the domain** | It was trained on wideband celebrity speech from VoxCeleb/VoxBlink. | Fine-tune the head (or LoRA) on helpline callers, using repeat-caller phone numbers as weak labels. |
| **Non-commercial licence** | The weights and VoxBlink2 data are CC BY-NC-SA. | For production, use a VoxCeleb-only model (e.g. WeSpeaker ResNet293, ReDimNet2-B6 vox2) fine-tuned on our own consented data. |
| **The web app has no login** | Anyone with the tunnel URL can enroll, verify and delete speakers. | Keep it on localhost or a private network, or put it behind authentication. |
