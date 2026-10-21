"""Speaker embeddings from w2v-BERT 2.0 LoRA_Adapter_MFA, large-margin fine-tuned (Li, Cheng, Li 2025,
arXiv 2510.04213; VoxCeleb2 + VoxBlink2; 0.14 % EER on Vox1-O). Weights: zl389/w2v-bert-2.0_SV model_lmft_0.14.pth.
Code: ZXHY-82/w2v-BERT-2.0_SV (cloned into repo/). Licence of code and weights: CC BY-NC-SA 4.0.

    from sv_model import SpeakerModel
    m = SpeakerModel()                       # loads on the GPU (about 3 GB in bf16)
    e1, e2 = m.embed_file("a.wav"), m.embed_file("b.wav")
    m.score(e1, e2)                          # cosine similarity, -1..1

Audio is decoded by ffmpeg to 16 kHz mono, so 8 kHz phone calls, mp3, webm etc. all work (8 kHz is upsampled).
"""
import os, subprocess, sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
CKPT = os.path.join(HERE, "models", "w2v-bert-2.0_SV", "model_lmft_0.14.pth")
SR = 16000
MAX_S = 60  # the authors' max_valid_dur: longer audio is cut to its first 60 s

# the repo's ecapa_tdnn.py imports calflops (FLOP counting, not used here); stub it rather than install it
if "calflops" not in sys.modules:
    import types
    sys.modules["calflops"] = types.SimpleNamespace(calculate_flops=None)

for p in (REPO, os.path.join(REPO, "recipes", "DeepASV"),
          os.path.join(REPO, "deeplab", "pretrained", "audio2vector", "module", "transformers", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def decode(data_or_path):
    """Any audio (path or bytes) -> float32 mono 16 kHz numpy array."""
    src = data_or_path if isinstance(data_or_path, str) else "pipe:0"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=None if isinstance(data_or_path, str) else data_or_path, capture_output=True)
    if r.returncode != 0:
        raise ValueError("could not decode audio: " + r.stderr.decode(errors="replace").strip()[-300:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


class SpeakerModel:
    def __init__(self, ckpt=CKPT, device="cuda"):
        from local.spk_model import Audio2Vec_based_Adapter  # the authors' model class (repo/recipes/DeepASV)

        # same arguments as the released train.yaml (stage 3, LMFT)
        self.model = Audio2Vec_based_Adapter(
            model_name="facebook/w2v-bert-2.0", frozen_encoder=False, bnb_config=None, peft_config=None,
            encoder_config="config_prune_tea.json", n_mfa_layers=-1, pooling_layer="ASP", embd_dim=256,
            adapter_dim=128, dropout=0.0)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["modules"]["spk_model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "hard_concrete" not in k]
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        self.unexpected = unexpected
        self.device = device
        self.model = self.model.eval().to(device)
        self.dtype = torch.bfloat16  # the authors scored with autocast bf16 (use_amp: true)

    @torch.inference_mode()
    def embed(self, wav):
        """float32 16 kHz mono array -> L2-normalised 256-d embedding (numpy)."""
        wav = np.asarray(wav, dtype=np.float32)[: MAX_S * SR]
        with torch.autocast("cuda", self.dtype):
            e = self.model(torch.from_numpy(wav)).float()  # 1-D input, as in the authors' get_embd_w2v.py
        return F.normalize(e, dim=-1)[0].cpu().numpy()

    def embed_file(self, data_or_path):
        return self.embed(decode(data_or_path))

    @staticmethod
    def score(e1, e2):
        e1, e2 = np.asarray(e1, np.float32), np.asarray(e2, np.float32)
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
