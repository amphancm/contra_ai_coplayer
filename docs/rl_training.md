# RL Training & Evaluation Guide

Contra AI Co-Player V5 includes a Gymnasium environment (`ContraCoPlayerEnv`), a Proximal Policy Optimization (PPO) training pipeline built on **Stable-Baselines3**, and an evaluation CLI to benchmark trained models against the `RuleAI` baseline.

## Prerequisites

Install the optional RL dependencies (`gymnasium`, `stable-baselines3`, `torch`):

```bash
pip install -e '.[rl]'
```

## Gymnasium Environment (`ContraCoPlayerEnv`)

Defined in `src/contra_coplayer/env.py`, `ContraCoPlayerEnv` trains Player 2 headlessly against a simulated Player 1 partner (`DiverseP1Partner`, `RuleP1Partner`, or `ScriptedP1Partner`).

### Bootstrap Save State

Instead of booting through the title screen on every episode reset, the environment loads a serialized save state (`states/stage1_2p.state` by default). If the file does not exist, `bootstrap_stage1_2p_state()` automatically boots `Contra (U)`, selects 2-Player mode, waits until Stage 1 `PLAYING` with both P1 and P2 active, and saves the state to disk.

### Action Space (`Discrete(15)`)

Each RL step repeats the selected action for `frame_skip` frames (default: `4` frames ≈ 15 decisions/sec at 60 FPS). When `B` (shoot) is included in an action, the environment (and `PPOPolicy` during live play) automatically pulses `B` on a 3-frames-on / 3-frames-off cadence so the default NES semi-auto rifle fires continuously. Indices `0..11` are preserved for backward compatibility with earlier models.

| Index | Name | Libretro Buttons | Behavior |
| --- | --- | --- | --- |
| `0` | `NOOP` | `{}` | Stand still |
| `1` | `RIGHT` | `{RIGHT}` | Walk right |
| `2` | `LEFT` | `{LEFT}` | Walk left |
| `3` | `RIGHT+B` | `{RIGHT, B}` | Run right + shoot |
| `4` | `LEFT+B` | `{LEFT, B}` | Run left + shoot |
| `5` | `RIGHT+A` | `{RIGHT, A}` | Jump right |
| `6` | `RIGHT+A+B` | `{RIGHT, A, B}` | Jump right + shoot |
| `7` | `LEFT+A+B` | `{LEFT, A, B}` | Jump left + shoot |
| `8` | `UP+RIGHT+B` | `{UP, RIGHT, B}` | Aim up-right + shoot |
| `9` | `DOWN+RIGHT+B` | `{DOWN, RIGHT, B}` | Aim down-right + shoot |
| `10` | `DOWN+B` | `{DOWN, B}` | Crouch / prone + shoot |
| `11` | `B` | `{B}` | Stand + shoot |
| `12` | `UP+B` | `{UP, B}` | Aim straight up + shoot |
| `13` | `DOWN+A` | `{DOWN, A}` | Drop down from platform |
| `14` | `A+B` | `{A, B}` | Jump in place + shoot |

### Observation Space (`Box(-1.0, 1.0, shape=(38 * frame_stack,), dtype=float32)`)

`encode_observation(state)` converts `GameState` into a normalized 38-dimensional vector. With `--frame-stack 4` (the default in `contra-train`), the last 4 RL-step observations are concatenated into a **152-dimensional** vector so the MLP policy can infer bullet/enemy velocities and jump trajectories. Both `PPOPolicy` and `contra-eval` automatically infer `frame_stack` from the loaded model's observation dimension (preserving compatibility with legacy 38-dim models).

| Indices | Count | Features | Normalization |
| --- | --- | --- | --- |
| `0..5` | 6 | P2 `x`, `y`, `active`, `lives`, `weapon_id`, `scroll_x` | `x/256`, `y/240`, `{0,1}`, `min(1, lives/5)`, `min(1, (wid&0xF)/8)`, `min(1, scroll_x/4096)` |
| `6..10` | 5 | P1 `x`, `y`, `active`, `p1_p2_dx`, `p1_p2_dy` | `x/256`, `y/240`, `{0,1}`, `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)` |
| `11..14` | 4 | P2 obstacles: `ground_ahead`, `pit_ahead`, `wall_ahead`, `platform_above` | `{0.0, 1.0}` |
| `15..26` | 12 | Nearest 4 enemies relative to P2: `(dx, dy, present)` × 4 | `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)`, `1.0` if present else `0.0` |
| `27..35` | 9 | Nearest 3 enemy bullets relative to P2: `(dx, dy, present)` × 3 | `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)`, `1.0` if present else `0.0` |
| `36..37` | 2 | Entity density: `len(enemies)`, `len(enemy_bullets)` | `min(1, count/16)`, `min(1, count/8)` |

### Co-Op Reward Shaping (`RewardConfig`)

`compute_reward(prev, curr, cfg, stagnant=...)` sums the following reward terms on every step:

| Term | Default Weight / Value | Condition |
| --- | --- | --- |
| `progress` | `+0.04` per px | `max(0, curr.scroll_x - prev.scroll_x)` |
| `p2_score` | `+0.004` per point | `max(0, curr.p2.score - prev.p2.score)` |
| `p1_score` | `+0.001` per point | `max(0, curr.p1.score - prev.p1.score)` |
| `survival` | `+0.015` | Each step P2 is active |
| `coop` (sync) | `+0.02` | Both active and `p1_p2_distance <= 85.0` px |
| `coop` (too far) | `-0.03` | Both active and `p1_p2_distance >= 115.0` px |
| `weapon` | `+4.0` | P2 active and picks up a new non-default weapon |
| `danger` | up to `-0.08` | P2 active and `nearest_threat_dist < 36.0` px; scaled as `-0.08 * (0.25 + 0.75 * (1 - dist/36))` |
| `stagnation` | `-0.02` | P2 active and `scroll_x` has not increased for `>= 30` consecutive steps |
| `death` | `-20.0` | `curr.p2.lives < prev.p2.lives` |
| `stage_clear` | `+40.0` | `curr.stage > prev.stage` |

### Simulated P1 Partner & Episode Termination

- **`--p1-partner` (`diverse` | `rule` | `scripted`)**:
  - `rule` (`RuleP1Partner`): Smart co-op Player 1 partner that detects pits/walls ahead, jumps over incoming enemy bullets and close enemies, fires continuously, and pauses briefly if it runs far ahead of P2.
  - `diverse` (`DiverseP1Partner`, default in `contra-train`): Randomly samples per episode among normal `RuleP1Partner` (60%), cautious `RuleP1Partner` (20%), and `ScriptedP1Partner` (20%) so P2 generalizes across different human playstyles.
  - `scripted` (`ScriptedP1Partner`): Original simple partner that advances right with random pauses/jumps.
  - You can also pass a custom `p1_policy` callable to `ContraCoPlayerEnv`.
- **`EvalCallback` (`--eval-freq`, `--eval-episodes`)**: Periodically evaluates the policy deterministically and saves the highest-scoring checkpoint to `<out_dir>/<out_stem>_best/best_model.zip` (along with matching `.json` metadata).
- **Termination**: Episode terminates on `GAME_OVER`, when P2 runs out of lives (`lives == 0` and inactive after at least one death), or immediately on P2 death if `terminate_on_p2_death=True`.
- **Truncation**: Episode truncates after `max_steps` RL steps (default `1500` steps = 6,000 frames at `frame_skip=4`).

## Training with `contra-train`

### Training From Scratch (V2 Baseline Architecture)

For a fresh model with 4-frame stacking (`152`-dim input) and a `256×256` MLP policy:

```bash
contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --out models/ppo_contra_v2 \
  --save-state states/stage1_2p.state \
  --timesteps 1000000 \
  --n-envs 10 \
  --n-steps 512 \
  --batch-size 256 \
  --lr 3e-4 \
  --ent-coef 0.015 \
  --net-arch 256,256 \
  --frame-skip 4 \
  --frame-stack 4 \
  --p1-partner diverse \
  --max-steps 1500 \
  --checkpoint-freq 250000 \
  --eval-freq 100000 \
  --eval-episodes 5 \
  --device cpu \
  --seed 42
```

Linear learning-rate decay is enabled by default (pass `--no-lr-decay` to keep a constant learning rate). Because `MlpPolicy` is lightweight, `cpu` with parallel `SubprocVecEnv` workers is often comparable to or faster than `cuda` for smaller networks, while `cuda` works well with `256,256` and `batch_size=256`.

### V3 Training (`ppo_contra_v3_rulep1`)

V3 fine-tunes from `models/ppo_contra_v2.zip` for an additional **1,000,000 timesteps** (2M total) with:

- **`--p1-partner diverse`**: trains P2 against a per-episode mix of normal `RuleP1Partner` (60%), cautious `RuleP1Partner` (20%), and `ScriptedP1Partner` (20%) instead of a single scripted partner.
- **`--frame-stack 4`**: preserves 4-frame stacked observations (`152`-dim).
- **`--lr 1e-4` (with linear decay) and `--ent-coef 0.01`**: lower fine-tuning learning rate while retaining exploration.
- **`--eval-freq 100000 --eval-episodes 5`**: runs deterministic `EvalCallback` evaluation every 100k steps and saves the highest-scoring checkpoint to `models/ppo_contra_v3_rulep1_best/best_model.zip`.

Run the bundled helper script:

```bash
./scripts/train_v3_rulep1.sh
```

Or invoke `contra-train` directly:

```bash
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
```

When `--resume` is used, `learning_rate` (including its linear decay schedule) and `ent_coef` are updated from the CLI flags and `reset_num_timesteps=False` preserves the total step counter.

### Outputs

- `<out>.zip`: Final Stable-Baselines3 PPO model.
- `<out>.json`: Training metadata summary (timesteps, elapsed time, steps/sec, hyperparameters, `frame_stack`, `p1_partner`, `eval_freq`, `best_model_path`, device, seed, resume source).
- `<out_dir>/<out_stem>_best/best_model.zip` & `best_model.json`: Highest-scoring deterministic evaluation model when `--eval-freq > 0`.
- `<out_dir>/checkpoints/<name>_<steps>_steps.zip`: Periodic checkpoints when `--checkpoint-freq > 0`.

### Reference Training Runs

| Run | Base / Resume | Timesteps | `n_envs` | `frame_stack` | `p1_partner` | Device | Throughput | Wall Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `models/ppo_contra.json` (V1) | From scratch (`128,128`) | 1,500,000 | 8 | 1 | `scripted` | `cpu` | ~704 steps/s (~2,816 fps) | ~2,131 s (~35.5 min) |
| `models/ppo_contra_v2.json` (V2) | From scratch (`256,256`) | 1,000,000 | 10 | 4 | `scripted` | `cpu` | ~560 steps/s (~2,240 fps) | ~1,785 s (~29.8 min) |
| `models/ppo_contra_v2_2m.json` (V2-2M) | `ppo_contra_v2.zip` | +1,000,000 | 10 | 4 | `scripted` | `cuda` | ~613 steps/s (~2,454 fps) | ~1,630 s (~27.2 min) |
| `models/ppo_contra_v3_rulep1.json` (V3) | `ppo_contra_v2.zip` | +1,000,000 | 10 | 4 | `diverse` | `cuda` | ~579 steps/s (~2,317 fps) | ~1,727 s (~28.8 min) |

## Evaluation with `contra-eval`

Benchmark a trained PPO model against the `RuleAI` baseline using the same seeds, save state, and simulated P1 partner (`--p1-partner scripted|rule|diverse`, default `scripted`):

```bash
# Evaluate V3 best model against RuleP1Partner (deterministic actions by default)
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --model models/ppo_contra_v3_rulep1_best/best_model.zip \
  --p1-partner rule \
  --episodes 5 \
  --max-steps 800

# RuleAI baseline against the same P1 partner
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --p1-partner rule \
  --episodes 5 \
  --max-steps 800
```

The report prints:
- **Mean Reward** (± std)
- **Mean Scroll X** (px)
- **Mean P2 Score**
- **Mean Steps**
- **Mean P2 Deaths**
- **Mean P1-P2 Dist** (px)

Add `--stochastic` to evaluate PPO with stochastic action sampling instead of deterministic argmax actions.
