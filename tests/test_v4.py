from pathlib import Path
import numpy as np
import pytest

from contra_coplayer.app import A, B, RIGHT, RuleAI
from contra_coplayer.dashboard import TerminalDashboard
from contra_coplayer.libretro import LibretroHost
from contra_coplayer.state import GameStateDetector, analyze_obstacles


def _make_ram(overrides: dict[int, int]):
    ram = bytearray(0x800)
    for addr, val in overrides.items():
        ram[addr] = val
    return bytes(ram)


def test_state_detector_ram_extraction():
    ram = _make_ram({
        0x22: 1,      # 2P mode
        0x2C: 4,      # PLAYING
        0x30: 0,      # Stage 1 (Jungle)
        0x32: 2,      # P1 lives
        0x33: 3,      # P2 lives
        0x64: 1,      # scroll high
        0x65: 76,     # scroll low -> 332
        0x90: 1,      # P1 active
        0x91: 1,      # P2 active
        0xAA: 4,      # P1 Spread
        0xAB: 0x11,   # P2 Machine Gun + Rapid
        0x334: 176,   # P1 X
        0x31A: 100,   # P1 Y
        0x335: 146,   # P2 X
        0x31B: 100,   # P2 Y
        0x7E2: 5,     # P1 score = 500
        0x7E4: 12,    # P2 score = 1200
        # Enemy slot 0: soldier at (210, 100)
        0x4B8: 0x03,
        0x33E: 210,
        0x324: 100,
        # Enemy slot 1: enemy bullet at (170, 100)
        0x4B9: 0x05,
        0x33F: 170,
        0x325: 100,
        # Player bullet slot 8 (P2): at (180, 95)
        0x390: 1,
        0x3D0: 180,
        0x3C0: 95,
    })
    detector = GameStateDetector()
    state = detector.detect(None, ram, frame_idx=42)

    assert state.phase == "PLAYING"
    assert state.stage == 1
    assert state.stage_name == "Jungle"
    assert state.two_player is True
    assert state.scroll_x == 332
    assert state.p1.active and (state.p1.x, state.p1.y) == (176, 100)
    assert state.p1.weapon_name == "Spread"
    assert state.p1.score == 500
    assert state.p2.active and (state.p2.x, state.p2.y) == (146, 100)
    assert state.p2.weapon_name == "Machine Gun+R"
    assert state.p2.score == 1200
    assert state.p1_p2_dx == -30.0
    assert state.p1_p2_dy == 0.0
    assert state.p1_p2_distance == 30.0
    assert len(state.enemies) == 1
    assert len(state.enemy_bullets) == 1
    assert len(state.player_bullets) == 1
    assert state.nearest_enemy_dist == pytest.approx(64.0)
    assert state.nearest_threat_dist == pytest.approx(24.0)


def test_analyze_obstacles_ground_vs_pit():
    # Solid textured ground below y=118
    frame_ground = np.zeros((224, 256, 3), dtype=np.uint8)
    frame_ground[118:132:2, :, :] = 180
    obs = analyze_obstacles(frame_ground, 100, 100)
    assert obs.ground_ahead is True
    assert obs.pit_ahead is False
    assert obs.ground_y is not None

    # Dark pit ahead-below
    frame_pit = np.zeros((224, 256, 3), dtype=np.uint8)
    obs_pit = analyze_obstacles(frame_pit, 100, 100)
    assert obs_pit.ground_ahead is False
    assert obs_pit.pit_ahead is True


def test_rule_ai_dodges_enemy_bullet_with_v4_state():
    ram = _make_ram({
        0x2C: 4, 0x90: 1, 0x91: 1,
        0x334: 100, 0x31A: 100,
        0x335: 120, 0x31B: 100,
        0x4B8: 0x05, 0x33E: 140, 0x324: 100,  # bullet 20px ahead of P2
    })
    state = GameStateDetector().detect(None, ram, frame_idx=10)
    ai = RuleAI(jump_period=100, jump_length=0)  # periodic jump disabled
    keys = ai.act(None, state)
    assert A in keys and B in keys and RIGHT in keys


def test_terminal_dashboard_render_text():
    ram = _make_ram({
        0x22: 1, 0x2C: 4, 0x30: 0, 0x32: 2, 0x33: 2,
        0x90: 1, 0x91: 1, 0x334: 150, 0x31A: 100, 0x335: 130, 0x31B: 100,
    })
    state = GameStateDetector().detect(None, ram, frame_idx=500)
    dash = TerminalDashboard(use_color=False)
    text = dash.render_text(state, actual_fps=60.0, target_fps=60.1, p1_keys={RIGHT, B}, p2_keys={RIGHT, B, A})
    assert "CONTRA AI CO-PLAYER" in text
    assert "Stage 1 - Jungle" in text
    assert "PLAYING" in text
    assert "2P Co-Op" in text
    assert "SYNCED" in text
    assert "Mini-Radar" in text


def test_live_contra_rom_v4_integration():
    root = Path(__file__).resolve().parents[1]
    core = root / "cores" / "fceumm_libretro.so"
    rom = root / "roms" / "Contra (U).nes"
    if not core.is_file() or not rom.is_file():
        pytest.skip("FCEUmm core or Contra ROM not present")

    detector = GameStateDetector()
    B_BTN, SELECT_BTN, START_BTN, RIGHT_BTN = 0, 2, 3, 7
    with LibretroHost(core, rom) as host:
        state = None
        for i in range(820):
            p1, p2 = set(), set()
            if i in (60, 61):
                p1.add(START_BTN)
            elif i in (80, 81):
                p1.add(SELECT_BTN)
            elif i in (100, 101):
                p1.add(START_BTN)
            elif i >= 740:
                p1 = {RIGHT_BTN, B_BTN} if i % 6 < 3 else {RIGHT_BTN}
                p2 = {RIGHT_BTN, B_BTN} if i % 8 < 4 else {RIGHT_BTN}
            frame, _ = host.step(p1, p2)
            if i == 819:
                state = detector.detect(frame, host.read_ram(), frame_idx=i)

    assert state is not None
    assert state.phase == "PLAYING"
    assert state.stage == 1
    assert state.stage_name == "Jungle"
    assert state.two_player is True
    assert state.p1.active and state.p2.active
    assert state.p1.x > 0 and state.p2.x > 0
    assert state.p1_p2_distance >= 0.0
