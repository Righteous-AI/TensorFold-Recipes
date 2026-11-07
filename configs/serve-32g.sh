#!/bin/bash
# TensorFold native CUDA server, Qwen3.8-27B EXL3 6.0bpw with DFlash2, port 12350.
# One process owns the GPU: stop the other engine units before starting this.
# The build does not start this script.
# Set TENSORFOLD_KEY to your own LAN token before starting; the fallback below is a
# placeholder, not a working secret. The bind is all interfaces, so keep it firewalled.
set -eu

ROOT=${TENSORFOLD_ROOT:-/opt/tensorfold/tensorfold}
BIN=$ROOT/zig-out/native/bin/tensorfold-native
MODEL=${TENSORFOLD_MODEL:-/mnt/models/qwen/Qwen3.8-27B-exl3-SC_6.00bpw_H6_V6}
DRAFT=${TENSORFOLD_DRAFTER:-/mnt/models/qwen/Qwen3.8-27B-DFlash2}
NAME=${TENSORFOLD_NAME:-qwen3.8-27b-exl3-6bpw}
PORT=${TENSORFOLD_PORT:-12350}
KEY=${TENSORFOLD_KEY:-change-me}
CONTEXT=${TENSORFOLD_CONTEXT:-262144}

for p in "$MODEL" "$DRAFT"; do
  if [ ! -d "$p" ]; then
    echo "no such directory: $p" >&2
    exit 2
  fi
done

if [ ! -x "$BIN" ]; then
  echo "no such binary: $BIN" >&2
  echo "build it: cd ${ROOT} && zig build -j10 -Dgpu=sm_120 -Dnvcc=\${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" >&2
  exit 2
fi

exec "$BIN" serve "$MODEL" \
  --name "$NAME" \
  --api-key "$KEY" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --backend cuda \
  --parallel 1 \
  --context "$CONTEXT" \
  --prompt-cache-gib 0 \
  --profile stable \
  --kv-cache fp8 \
  --drafter "$DRAFT"

# Notes on the flags, all confirmed against v1.2.3:
#   --kv-cache        bf16 or fp8 only. nvfp4 and fp4 exit 2. fp8 is what leaves room
#                     for the 262144 window; serve --help does not list the flag.
#   --drafter         must be the BF16 DFlash2 directory. The loader quantizes its
#                     projections to affine-4, group 64, at load. A pre-quantized
#                     drafter on disk is refused. --drafter-bits stays at 4, its default.
#   --context         262144 is the model's max_position_embeddings. At this window the
#                     engine maps the bf16 embedding rows into pinned host memory; a
#                     lower window keeps them on the card.
#   --parallel        1. A stream costs 9.09 GiB at this window and only one fits.
#   --prompt-cache-gib 0  no retained prefix: every prompt pays its own prefill.
