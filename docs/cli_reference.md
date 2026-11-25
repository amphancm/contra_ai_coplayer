# CLI & Controls Reference

Installing the package exposes three console scripts (configured in `pyproject.toml`):

- `contra-coplayer` (also `python -m contra_coplayer`) — Interactive Libretro frontend with P2 AI
- `contra-train` (also `python -m contra_coplayer.train`) — PPO RL training pipeline
- `contra-eval` (also `python -m contra_coplayer.eval_model`) — Policy evaluation & comparison

---

## `contra-coplayer`

Launch the live Contra game window with P1 human controls, P2 AI, SDL2 audio, and the terminal dashboard.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--core` | `str` | *(required)* | Path to `fceumm_libretro.so` |
| `--rom` | `str` | *(required)* | Path to legally obtained Contra NES ROM |
| `--system-dir` | `str` | ROM parent dir | Libretro system directory |
| `--save-dir` | `str` | ROM parent dir | Libretro save directory |
| `--scale` | `int` | `3` | Window scale factor (base resolution `256×240`) |
| `--frames` | `int` | `0` | Stop after `N` frames (`0` = run until quit) |
| `--no-audio` | flag | `False` | Disable SDL2 audio output |
| `--no-ai` | flag | `False` | Disable P2 AI (`ai=False`) |
| `--model` | `str` | `None` | Path to trained PPO `.zip` model for P2 AI (uses `RuleAI` if omitted) |
| `--auto-start` | flag | `False` | Press `START` on both P1 and P2 for the first 2 frames |
| `--auto-2p` | flag | `False` | Auto-select 2-Player mode on the Contra title screen (Start @ 60–61, Select @ 80–81, Start @ 100–101) |
| `--no-dashboard` | flag | `False` | Disable live ANSI terminal dashboard |
| `--dashboard-hz` | `float` | `10.0` | Terminal dashboard refresh rate in Hz |
| `--debug-overlay` | flag | `False` | Draw V4 detection boxes and P1–P2 tether line on the game window |

---

## `contra-train`

Train a Player 2 PPO policy headlessly in `ContraCoPlayerEnv`.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--core` | `str` | *(required)* | Path to `fceumm_libretro.so` |
| `--rom` | `str` | *(required)* | Path to Contra NES ROM |
| `--out` | `str` | `models/ppo_contra` | Output model path prefix (saves `.zip` and `.json`) |
| `--save-state` | `str` | `states/stage1_2p.state` | Bootstrap save-state file (auto-created if missing) |
| `--timesteps` | `int` | `50000` | Total training timesteps |
| `--n-envs` | `int` | `1` | Number of environments (`SubprocVecEnv` when `> 1`, `DummyVecEnv` when `1`) |
| `--n-steps` | `int` | `512` | Rollout steps per environment per PPO update |
| `--batch-size` | `int` | `64` | Minibatch size |
| `--lr` | `float` | `3e-4` | Initial learning rate |
| `--no-lr-decay` | flag | `False` | Disable linear learning-rate decay |
| `--ent-coef` | `float` | `0.015` | Entropy coefficient |
| `--net-arch` | `str` | `128,128` | Comma-separated MLP hidden layer sizes (e.g. `256,256`) |
| `--frame-skip` | `int` | `4` | Emulator frames per RL action |
| `--frame-stack` | `int` | `4` | Number of stacked observations per RL step (`38 * frame_stack` dims) |
| `--max-steps` | `int` | `1500` | Max RL steps per episode |
| `--p1-partner` | `str` | `diverse` | Simulated P1 partner policy (`diverse`, `rule`, or `scripted`) |
| `--terminate-on-death` | flag | `False` | Terminate episode immediately on P2 death |
| `--device` | `str` | `auto` | PyTorch device (`auto`, `cpu`, `cuda`) |
| `--seed` | `int` | `42` | Random seed |
| `--checkpoint-freq` | `int` | `0` | Checkpoint interval in total timesteps (`0` = disabled) |
| `--eval-freq` | `int` | `0` | `EvalCallback` interval in total timesteps (`0` = disabled); saves best model to `<out>_best/best_model.zip` |
| `--eval-episodes` | `int` | `5` | Number of deterministic episodes per `EvalCallback` evaluation |
| `--resume` | `str` | `None` | Path to existing PPO `.zip` model to continue training |

---

## `contra-eval`

Evaluate a trained PPO model or the `RuleAI` baseline across multiple episodes.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--core` | `str` | *(required)* | Path to `fceumm_libretro.so` |
| `--rom` | `str` | *(required)* | Path to Contra NES ROM |
| `--model` | `str` | `None` | Path to trained PPO `.zip` model (omit to evaluate `RuleAI` baseline) |
| `--save-state` | `str` | `states/stage1_2p.state` | Bootstrap save-state file |
| `--episodes` | `int` | `5` | Number of evaluation episodes |
| `--max-steps` | `int` | `800` | Max RL steps per episode |
| `--frame-skip` | `int` | `4` | Emulator frames per RL action |
| `--p1-partner` | `str` | `scripted` | Simulated P1 partner policy for evaluation (`scripted`, `rule`, or `diverse`) |
| `--stochastic` | flag | `False` | Sample stochastic PPO actions instead of deterministic argmax |
| `--seed` | `int` | `123` | Base random seed (`seed + ep` per episode) |

---

## Controls (Player 1)

### Keyboard

Both keysyms and physical USB HID scancodes are mapped, so keys work regardless of active OS keyboard layout.

| NES Button | Keys |
| --- | --- |
| D-Pad (`UP` / `DOWN` / `LEFT` / `RIGHT`) | Arrow keys |
| `B` (Shoot) | `Z`, `J` |
| `A` (Jump) | `X`, `K`, `Space` |
| `START` | `Enter` |
| `SELECT` | `Right Shift` |
| Quit | `Escape` |

### Gamepad

Uses the first connected SDL joystick (supports hot-plugging during gameplay):

| NES Button | Gamepad Mapping |
| --- | --- |
| D-Pad | Left analog stick (axis 0/1, threshold `±0.45`) or D-Pad (hat 0) |
| `B` (Shoot) | Button `0` |
| `A` (Jump) | Button `1` |
| `SELECT` | Button `6` |
| `START` | Button `7` |

---

## Live Terminal Dashboard Legend

The ANSI dashboard updates in-place when `stdout` is a TTY (disable with `--no-dashboard`).

### Co-Op Status Tags

| Tag | Meaning |
| --- | --- |
| `SYNCED` | Both P1 and P2 active, within `±45` px horizontally and `<= 110` px apart |
| `P2 AHEAD` | P2 is more than `45` px ahead of P1 (`p1_p2_dx > 45`) |
| `P2 BEHIND` | P2 is more than `45` px behind P1 (`p1_p2_dx < -45`) |
| `TOO FAR` | P1–P2 Euclidean distance exceeds `110` px |
| `WAITING` | One or both players are inactive (title screen, intro, respawn) |

### Mini-Radar Symbols (`256×224` projected onto `40×7`)

| Symbol | Entity |
| --- | --- |
| `1` | Player 1 (Human) |
| `2` | Player 2 (AI) |
| `E` | Enemy |
| `*` | Enemy bullet |
| `.` | Player bullet |

### On-Screen Debug Overlay (`--debug-overlay`)

| Visual | Entity |
| --- | --- |
| Green rectangle | Player 1 (Human) |
| Cyan rectangle | Player 2 (AI) |
| Yellow line | P1–P2 distance vector |
| Red rectangle | Enemy |
| Orange circle | Enemy bullet |
| White circle | Player bullet |
