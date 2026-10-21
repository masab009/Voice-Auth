#!/usr/bin/env bash
# Fetch everything the demo needs that is not in this repository:
#   1. the authors' code (ZXHY-82/w2v-BERT-2.0_SV, pinned commit) into repo/  -- model classes + their transformers fork
#   2. the speaker-verification checkpoint model_lmft_0.14.pth (2.35 GB) from Hugging Face zl389/w2v-bert-2.0_SV
#      into models/w2v-bert-2.0_SV/, resumable, checked by size and SHA-256.
# The facebook/w2v-bert-2.0 base weights are NOT needed: the checkpoint holds the whole fine-tuned encoder, and the
# feature-extractor config ships inside the authors' repo.
#
#   bash download_models.sh            # safe to re-run; skips what is already complete
#   HF_TOKEN=... bash download_models.sh   # optional, only if Hugging Face rate-limits anonymous downloads
set -euo pipefail
cd "$(dirname "$0")"

REPO_URL=https://github.com/ZXHY-82/w2v-BERT-2.0_SV.git
REPO_COMMIT=98ff0e29c41d22f8164f4ac5bff7f21cc5d9a95d
HF_URL=https://huggingface.co/zl389/w2v-bert-2.0_SV/resolve/main
CKPT=models/w2v-bert-2.0_SV/model_lmft_0.14.pth
CKPT_SIZE=2353147380
CKPT_SHA256=454398fe4010bb7d1095517f3aff683c90a1eed644f68f83272c38ea81f953f9

# 1. authors' code
if [ ! -d repo/.git ]; then
  echo "== cloning $REPO_URL"
  git clone --quiet "$REPO_URL" repo
fi
git -C repo fetch --quiet --depth 1 origin "$REPO_COMMIT" 2>/dev/null || true
git -C repo checkout --quiet "$REPO_COMMIT"
test -f repo/deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/config_prune_tea.json
test -f repo/deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/preprocessor_config.json
echo "repo/ at $(git -C repo rev-parse --short HEAD)"

# 2. checkpoint + the authors' training configs (for reference)
mkdir -p models/w2v-bert-2.0_SV/config/v1
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
for f in train.yaml config/v1/s1.yaml config/v1/s2.yaml config/v1/s3.yaml; do
  [ -s "models/w2v-bert-2.0_SV/$f" ] || curl -sfL "${auth[@]}" -o "models/w2v-bert-2.0_SV/$f" "$HF_URL/$f"
done
size() { stat -c %s "$1" 2>/dev/null || echo 0; }
for attempt in 1 2 3 4 5; do
  [ "$(size "$CKPT")" = "$CKPT_SIZE" ] && break
  echo "== downloading $(basename "$CKPT") (attempt $attempt, have $(size "$CKPT") of $CKPT_SIZE bytes)"
  curl -fL "${auth[@]}" -C - --retry 3 -o "$CKPT" "$HF_URL/$(basename "$CKPT")" || sleep 5
done
[ "$(size "$CKPT")" = "$CKPT_SIZE" ] || { echo "download incomplete: $CKPT"; exit 1; }
echo "== checking SHA-256 (about a minute)"
echo "$CKPT_SHA256  $CKPT" | sha256sum -c -
echo "done. Start the demo with: bash run.sh"
