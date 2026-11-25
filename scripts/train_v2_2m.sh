#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --save-state states/stage1_2p.state \
  --resume models/ppo_contra_v2.zip \
  --out models/ppo_contra_v2_2m \
  --timesteps 1000000 \
  --n-envs 10 \
  --batch-size 256 \
  --lr 1e-4 \
  --ent-coef 0.008 \
  --frame-stack 4 \
  --checkpoint-freq 250000 \
  --device cuda
