#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --save-state states/stage1_2p.state \
  --resume models/ppo_contra_v2.zip \
  --out models/ppo_contra_v3_rulep1 \
  --timesteps 1000000 \
  --n-envs 10 \
  --batch-size 256 \
  --lr 1e-4 \
  --ent-coef 0.01 \
  --frame-stack 4 \
  --p1-partner diverse \
  --checkpoint-freq 250000 \
  --eval-freq 100000 \
  --eval-episodes 5 \
  --device cuda
