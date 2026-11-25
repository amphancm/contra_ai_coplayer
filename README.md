# Contra AI Co-Player

A Python Libretro frontend and AI co-player for NES **Contra** on Linux and WSL2. Play as **Player 1 (Human)** alongside an autonomous **Player 2 (AI)** powered by either a RAM/vision-aware rule-based policy (`RuleAI`) or a trained **PPO reinforcement learning** model (`PPOPolicy`).

## Features

- **Direct Libretro Core Host**: Loads `fceumm_libretro.so` directly via `ctypes` with software RGB frame decoding (`0RGB1555`, `RGB565`, `XRGB8888`), queued SDL2 stereo PCM audio, direct 2KB NES System RAM access, and binary save-state serialization.
- **SDL2 / Pygame Display & Input**: Resizable game window, layout-independent USB HID keyboard scancode input (works seamlessly with non-English keyboard layouts such as Thai), and hot-pluggable USB gamepad support.
- **V4 Game State Detection**: Combines direct NES RAM extraction (players, lives, weapons, scores, scroll position, enemies, enemy/player bullets, stage & phase) with lightweight RGB frame heuristics (ground, pits, walls, overhead platforms).
- **Live Terminal Dashboard & Debug Overlay**: Zero-dependency ANSI terminal HUD with real-time FPS, P1/P2 status, co-op spacing telemetry, entity counts, obstacle cues, and a 2D ASCII mini-radar, plus optional on-screen bounding-box overlay (`--debug-overlay`).
- **V5 PPO Reinforcement Learning**: Custom Gymnasium environment (`ContraCoPlayerEnv`) with a 38-dim observation vector, 12 discrete actions, shaped co-op reward function, parallel `SubprocVecEnv` training (`contra-train`), checkpointing/resuming, and baseline comparison (`contra-eval`).

## Requirements

- Ubuntu Linux or Ubuntu on WSL2 with WSLg enabled
- Python >= 3.10
- SDL2 runtime (`libsdl2-2.0-0`)
- A compatible `fceumm_libretro.so` Libretro core
- A legally obtained Contra (U) NES ROM (not bundled)

## Installation

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libsdl2-2.0-0

python3 -m venv .venv
source .venv/bin/activate

# Core frontend + tests
pip install -e '.[dev]'

# Optional: install RL dependencies (Gymnasium, Stable-Baselines3, PyTorch)
pip install -e '.[rl]'

# Or install everything
pip install -e '.[dev,rl]'
```

If you need to locate an existing FCEUmm core on your machine:

```bash
find /opt/retropie /usr/lib ~/.config/retroarch -name '*fceumm*libretro.so' 2>/dev/null
```

## Quick Start

### Play with Rule-Based P2 AI

```bash
contra-coplayer \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --auto-2p
```

*(You can also invoke via `python -m contra_coplayer ...`.)*

### Play with Trained PPO P2 AI

```bash
contra-coplayer \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --model models/ppo_contra.zip \
  --auto-2p \
  --debug-overlay
```

### Useful Runtime Flags

| Flag | Description |
| --- | --- |
| `--model PATH` | Use a trained Stable-Baselines3 PPO `.zip` model for P2 (defaults to `RuleAI` if omitted) |
| `--auto-2p` | Automatically navigate the Contra title screen into 2-Player mode |
| `--auto-start` | Press Start for both ports on the first two frames |
| `--debug-overlay` | Draw P1/P2/enemy/bullet boxes and P1–P2 tether line on the game window |
| `--no-dashboard` | Disable the live ANSI terminal dashboard |
| `--dashboard-hz N` | Set terminal dashboard refresh rate (default: `10.0` Hz) |
| `--no-audio` | Disable SDL2 audio output (useful in headless or audio-less environments) |
| `--no-ai` | Disable P2 AI completely |
| `--scale N` | Initial window scale factor over 256×240 (default: `3`) |
| `--frames N` | Stop automatically after `N` frames (`0` = run until quit) |

See [docs/cli_reference.md](docs/cli_reference.md) for the complete list of options across all commands.

## Controls (Player 1)

Keyboard input uses both keysyms and physical USB HID scancodes so controls remain responsive regardless of active OS keyboard layout.

| Action | Keyboard | Gamepad (1st Joystick) |
| --- | --- | --- |
| Move / Aim | Arrow Keys | Left Analog Stick / D-Pad (Hat 0) |
| Shoot (`B`) | `Z` or `J` | Button 0 (South) |
| Jump (`A`) | `X`, `K`, or `Space` | Button 1 (East) |
| Start | `Enter` | Button 7 |
| Select | `Right Shift` | Button 6 |
| Quit | `Escape` or close window | — |

## Training & Evaluating PPO AI

Install the `[rl]` extras first (`pip install -e '.[rl]'`).

### Train a PPO Model

```bash
contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --out models/ppo_contra \
  --save-state states/stage1_2p.state \
  --timesteps 1500000 \
  --n-envs 8 \
  --batch-size 256 \
  --checkpoint-freq 300000 \
  --device cpu
```

To resume training from an existing checkpoint:

```bash
contra-train \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --resume models/ppo_contra.zip \
  --out models/ppo_contra_v2 \
  --timesteps 500000 \
  --n-envs 8 \
  --device cpu
```

### Evaluate PPO vs. RuleAI Baseline

```bash
# Evaluate trained PPO model
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --model models/ppo_contra.zip \
  --episodes 5

# Evaluate RuleAI baseline (omit --model)
contra-eval \
  --core cores/fceumm_libretro.so \
  --rom 'roms/Contra (U).nes' \
  --episodes 5
```

For observation space, action space, reward shaping, and hyperparameter details, see [docs/rl_training.md](docs/rl_training.md).

## Project Structure

```text
contra_ai_coplayer/
├── pyproject.toml                  # Package metadata, dependencies, CLI entry points
├── README.md
├── docs/
│   ├── architecture.md             # System architecture, NES RAM map, and detection heuristics
│   ├── rl_training.md              # Gymnasium environment, reward shaping, and PPO training guide
│   └── cli_reference.md            # Complete CLI flags, controls, dashboard & overlay legends
├── src/contra_coplayer/
│   ├── __init__.py
│   ├── __main__.py                 # python -m contra_coplayer entry point
│   ├── libretro.py                 # Libretro C-API wrapper (video, audio, input, RAM, save states)
│   ├── state.py                    # V4 GameStateDetector (NES RAM + RGB obstacle analysis)
│   ├── app.py                      # Main pygame/SDL2 loop, RuleAI, input, debug overlay
│   ├── dashboard.py                # Live ANSI terminal dashboard & ASCII mini-radar
│   ├── env.py                      # V5 ContraCoPlayerEnv (Gymnasium), reward, ScriptedP1Partner
│   ├── policy.py                   # PPOPolicy wrapper for live gameplay inference
│   ├── train.py                    # Parallel PPO training pipeline & checkpointing
│   └── eval_model.py               # Policy evaluation & comparison CLI
├── tests/
│   ├── test_core.py                # Frame decoding, RuleAI, scancode mapping tests
│   ├── test_v4.py                  # RAM state detection, obstacle heuristics, dashboard, ROM tests
│   └── test_v5.py                  # Obs encoding, reward, save states, PPO train/eval smoke tests
├── cores/                          # Local Libretro core (e.g., fceumm_libretro.so)
├── roms/                           # Local NES ROMs (git-ignored)
├── states/                         # Bootstrap save states (git-ignored)
└── models/                         # Trained PPO models, JSON metadata, and checkpoints
```

## Tests

Run the unit and integration test suite:

```bash
pytest -q
```

Integration tests that require `cores/fceumm_libretro.so` and `roms/Contra (U).nes` automatically skip if those files are not present.

## Documentation

- [Architecture & State Detection](docs/architecture.md) — Runtime pipeline, module breakdown, `Contra (U)` NES RAM map, and RGB obstacle heuristics.
- [RL Training & Evaluation Guide](docs/rl_training.md) — `ContraCoPlayerEnv` design, 38-dim observation vector, 12-action space, co-op reward shaping, and PPO tuning.
- [CLI & Controls Reference](docs/cli_reference.md) — All CLI flags for `contra-coplayer`, `contra-train`, and `contra-eval`, plus dashboard and overlay legends.

## Scope & Limitations

- **ROM & Region Specificity**: RAM offsets in `GameStateDetector` are mapped for `Contra (U).nes`. Other regions or ROM hacks may differ in memory layout.
- **RL Training Scope**: The default bootstrap state starts in Stage 1 (Jungle) 2-Player mode with a scripted P1 partner (`ScriptedP1Partner`). Generalization to later stages or very different human playstyles may require additional save states and fine-tuning.
- **Audio Sync**: Libretro audio batch callbacks are copied immediately and fed to SDL2 queued audio with a ~250 ms latency cap rather than sample-accurate resampling.
- **Frame Pacing**: Uses `retro_get_system_av_info()` FPS and `time.perf_counter()`; late frames are not skipped.
- **Platform**: Targets Linux and WSL2 (little-endian signed 16-bit stereo PCM for SDL2 audio). USB gamepad passthrough on WSL2 depends on host USBIP setup; keyboard works out of the box.
- **Core Compatibility**: Implements the core Libretro environment callbacks required by FCEUmm; other cores may require extending callbacks in `libretro.py`.
- **Legal**: No copyrighted game ROMs are distributed with this repository. Supply your own legally acquired ROM.
