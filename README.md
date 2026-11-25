# Contra AI Co-Player V3 (Python Libretro Frontend)

A prototype Linux/WSL2 Libretro frontend with direct frame callbacks, SDL2-backed pygame display and joystick, queued SDL2 audio, and a basic rule-based P2 AI.

## Requirements

- Ubuntu Linux or Ubuntu on WSL2 with WSLg enabled
- A compatible `fceumm_libretro.so` and a legally acquired Contra NES ROM
- SDL2 runtime, Python >=3.10

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libsdl2-2.0-0
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
find /opt/retropie /usr/lib ~/.config/retroarch -name '*fceumm*libretro.so' 2>/dev/null
python -m contra_coplayer --core /absolute/path/fceumm_libretro.so --rom '/absolute/path/Contra (U).nes' --auto-start
```

P1 keyboard: arrows = movement, Z = B (shoot), X = A (jump), Enter = Start, Right Shift = Select. Physical gamepad: first joystick, left analog/D-pad, buttons 0/1, 6/7. Actual mappings depend on the device. P2: AI holds shoot/right and jumps periodically. Escape exits. Use `--no-audio` when no SDL2 output device is available; `--no-ai` to disable P2. `--frames 600` stops after 600 frames.

## Tests

```bash
pytest -q
```

## Scope and limitations

- This project is a functional *prototype*, not yet verified on your physical hardware or actual Contra ROM.
- The P2 policy is only a baseline and cannot reliably clear stages; no OpenCV enemy detection or trained PPO yet.
- Libretro audio batch callbacks are copied immediately. SDL2 queued audio is kept within a rough latency cap, not sample-accurate sync.
- Frame scheduler uses `retro_get_system_av_info()` FPS and `time.perf_counter()`; late frames are not skipped.
- Basic libretro environment callbacks cover common FCEUmm requirements; newer cores may request unsupported commands. Inspect core logs and extend callbacks as necessary.
- Core options and runtime video geometry changes are only minimally supported.
- A libretro core and game ROM are not bundled.
- On WSL2, USB gamepad passthrough is environment-dependent; keyboard is the fallback.
- Linux only: SDL2 audio device uses little-endian signed 16-bit stereo PCM.
- A game may require choosing two-player mode in its own menu; auto-start does not guarantee P2 is activated.
