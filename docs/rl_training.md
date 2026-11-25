# RL Training & Evaluation Guide

Contra AI Co-Player V5 includes a Gymnasium environment (`ContraCoPlayerEnv`), a Proximal Policy Optimization (PPO) training pipeline built on **Stable-Baselines3**, and an evaluation CLI to benchmark trained models against the `RuleAI` baseline.

## Prerequisites

Install the optional RL dependencies (`gymnasium`, `stable-baselines3`, `torch`):

```bash
pip install -e '.[rl]'
```

## Gymnasium Environment (`ContraCoPlayerEnv`)

Defined in `src/contra_coplayer/env.py`, `ContraCoPlayerEnv` trains Player 2 headlessly against a simulated Player 1 partner (`ScriptedP1Partner`).

### Bootstrap Save State

Instead of booting through the title screen on every episode reset, the environment loads a serialized save state (`states/stage1_2p.state` by default). If the file does not exist, `bootstrap_stage1_2p_state()` automatically boots `Contra (U)`, selects 2-Player mode, waits until Stage 1 `PLAYING` with both P1 and P2 active, and saves the state to disk.

### Action Space (`Discrete(12)`)

Each RL step repeats the selected action for `frame_skip` frames (default: `4` frames ≈ 15 decisions/sec at 60 FPS). When `B` (shoot) is included in an action, the environment (and `PPOPolicy` during live play) automatically pulses `B` on a 3-frames-on / 3-frames-off cadence so the default NES semi-auto rifle fires continuously.

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

### Observation Space (`Box(-1.0, 1.0, shape=(38,), dtype=float32)`)

`encode_observation(state)` converts `GameState` into a normalized 38-dimensional vector:

| Indices | Count | Features | Normalization |
| --- | --- | --- | --- |
| `0..5` | 6 | P2 `x`, `y`, `active`, `lives`, `weapon_id`, `scroll_x` | `x/256`, `y/240`, `{0,1}`, `min(1, lives/5)`, `min(1, (wid&0xF)/8)`, `min(1, scroll_x/4096)` |
| `6..10` | 5 | P1 `x`, `y`, `active`, `p1_p2_dx`, `p1_p2_dy` | `x/256`, `y/240`, `{0,1}`, `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)` |
| `11..14` | 4 | P2 obstacles: `ground_ahead`, `pit_ahead`, `wall_ahead`, `platform_above` | `{0.0, 1.0}` |
| `15..26` | 12 | Nearest 4 enemies relative to P2: `(dx, dy, present)` × 4 | `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)`, `1.0` if present else `0.0` |
| `27..35` | 9 | Nearest 3 enemy bullets relative to P2: `(dx, dy, present)` × 3 | `clip(dx/256, -1, 1)`, `clip(dy/240, -1, 1)`, `1.0` if present else `0.0` |
| `36..37` | 2 | Entity density: `len(enemies)`, `len(enemy_bullets)` | `min(1, count/16)`, `min(1, count/8)` |

### Co-Op Reward Shaping (`RewardConfig`)

`compute_reward(prev, curr, cfg)` sums the following reward terms on every step:

| Term | Default Weight / Value | Condition |
| --- | --- | --- |
| `progress` | `+0.04` per px | `max(0, curr.scroll_x - prev.scroll_x)` |
| `p2_score` | `+0.02` per point | `max(0, curr.p2.score - prev.p2.score)` |
| `p1_score` | `+0.005` per point | `max(0, curr.p1.score - prev.p1.score)` |
| `survival` | `+0.01` | Each step P2 is active |
| `coop` (sync) | `+0.02` | Both active and `p1_p2_distance <= 85.0` px |
| `coop` (too far) | `-0.03` | Both active and `p1_p2_distance >= 115.0` px |
| `weapon` | `+3.0` | P2 active and picks up a new non-default weapon |
| `danger` | `-0.04` | P2 active and `nearest_threat_dist < 18.0` px |
| `death` | `-6.0` | `curr.p2.lives < prev.p2.lives` |
| `stage_clear` | `+25.0` | `curr.stage > prev.stage` |

### Simulated P1 Partner & Episode Termination

- **`ScriptedP1Partner`**: Advances right with occasional random pauses (10–35 frames), periodic rifle fire, and occasional random jumps (10–18 frames). You can pass a custom `p1_policy` callable to `ContraCoPlayerEnv` to train against different partner behaviors.
- **Termination**: Episode terminates on `GAME_OVER`, when P2 runs out of lives (`lives == 0` and inactive after at least one death), or immediately on P2 death if `terminate_on_p2_death=True`.
- **Truncation**: Episode truncates after `max_steps` RL steps (default `1500` steps = 6,000 frames at `frame_skip=4`).

## Training with `contra-train`

### Recommended Parallel Training (CPU)

Because the policy is a lightweight MLP (`MlpPolicy`, default `net_arch=[128, 128]`), training on **CPU** with parallel environments (`SubprocVecEnv`) is typically faster than GPU:

```bash
contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --out models/ppo_contra \
  --save-state states/stage1_2p.state \
  --timesteps 1500000 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --lr 3e-4 \
  --ent-coef 0.015 \
  --net-arch 128,128 \
  --frame-skip 4 \
  --max-steps 1500 \
  --checkpoint-freq 300000 \
  --device cpu \
  --seed 42
```

### Outputs

- `<out>.zip`: Final Stable-Baselines3 PPO model.
- `<out>.json`: Training metadata summary (timesteps, elapsed time, steps/sec, hyperparameters, device, seed, resume source).
- `<out_dir>/checkpoints/<name>_<steps>_steps.zip`: Periodic checkpoints when `--checkpoint-freq > 0`.

### Reference Training Run (`models/ppo_contra.json`)

The bundled reference configuration trained for **1,500,000 timesteps** with `n_envs=8`, `n_steps=512`, `batch_size=256`, `lr=3e-4`, `ent_coef=0.015`, `net_arch=[128, 128]`, `frame_skip=4`, on `cpu`:

- **Throughput**: ~704 steps/sec (~2,816 emulator frames/sec at `frame_skip=4`)
- **Total wall time**: ~2,131 seconds (~35.5 minutes)

### Resuming Training

```bash
contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --resume models/ppo_contra.zip \
  --out models/ppo_contra_resumed \
  --timesteps 500000 \
  --n-envs 8 \
  --lr 1e-4 \
  --device cpu
```

When `--resume` is used, `learning_rate` and `ent_coef` are updated from the CLI flags and `reset_num_timesteps=False` preserves the total step counter.

## Evaluation with `contra-eval`

Benchmark a trained PPO model against the `RuleAI` baseline using the same seeds and save state:

```bash
# PPO model (deterministic actions by default)
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --model models/ppo_contra.zip \
  --episodes 5 \
  --max-steps 800

# RuleAI baseline
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
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
