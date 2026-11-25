"""Real-time ANSI Terminal Dashboard for Contra AI Co-Player V4."""
from __future__ import annotations

import sys
import time
from typing import Iterable, TextIO

from .state import GameState

BUTTON_NAMES = {
    0: "B",
    1: "Y",
    2: "SEL",
    3: "STA",
    4: "UP",
    5: "DN",
    6: "LT",
    7: "RT",
    8: "A",
    9: "X",
    10: "L",
    11: "R",
}

# ANSI color codes
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"
BLUE = "\x1b[34m"


def format_buttons(buttons: Iterable[int]) -> str:
    names = [BUTTON_NAMES.get(b, str(b)) for b in sorted(set(buttons))]
    return "+".join(names) if names else "-"


def _coop_tag(state: GameState) -> str:
    if not (state.p1.active and state.p2.active):
        return "WAITING"
    if state.p1_p2_distance > 110:
        return "TOO FAR"
    if state.p1_p2_dx < -45:
        return "P2 BEHIND"
    if state.p1_p2_dx > 45:
        return "P2 AHEAD"
    return "SYNCED"


def _render_radar(state: GameState, width: int = 40, height: int = 7) -> list[str]:
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def plot(x: int, y: int, ch: str) -> None:
        gx = max(0, min(width - 1, int(x * width / 256)))
        gy = max(0, min(height - 1, int(y * height / 240)))
        grid[gy][gx] = ch

    for b in state.player_bullets:
        plot(b.x, b.y, ".")
    for e in state.enemies:
        plot(e.x, e.y, "E")
    for eb in state.enemy_bullets:
        plot(eb.x, eb.y, "*")
    if state.p1.active:
        plot(state.p1.x, state.p1.y, "1")
    if state.p2.active:
        plot(state.p2.x, state.p2.y, "2")

    return ["".join(row) for row in grid]


class TerminalDashboard:
    """Throttled, zero-dependency live terminal dashboard using ANSI escape sequences."""

    def __init__(
        self,
        stream: TextIO | None = None,
        refresh_hz: float = 10.0,
        use_color: bool | None = None,
        enabled: bool = True,
    ):
        self.stream = stream or sys.stdout
        self.refresh_interval = 1.0 / max(1.0, refresh_hz)
        self.enabled = enabled
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.use_color = self.is_tty if use_color is None else use_color
        self._last_draw = 0.0
        self._started = False

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.use_color else text

    def render_text(
        self,
        state: GameState,
        *,
        actual_fps: float = 0.0,
        target_fps: float = 60.0,
        p1_keys: Iterable[int] = (),
        p2_keys: Iterable[int] = (),
        audio_on: bool = True,
        ai_on: bool = True,
        ai_label: str | None = None,
        overlay_on: bool = False,
        input_device: str = "Keyboard",
    ) -> str:
        w = 76
        hr = "─" * w
        top = f"┌{hr}┐"
        mid = f"├{hr}┤"
        bot = f"└{hr}┘"

        def row(content: str, raw_len: int | None = None) -> str:
            visible = len(content) if raw_len is None else raw_len
            pad = max(0, w - 2 - visible)
            return f"│ {content}{' ' * pad} │"

        title = "CONTRA AI CO-PLAYER  |  V4 LIVE TERMINAL DASHBOARD"
        mode_str = "2P Co-Op" if state.two_player else "1P Mode"
        stage_line = (
            f"Stage   : Stage {state.stage} - {state.stage_name:<12}  "
            f"Phase: {state.phase:<11}  Mode: {mode_str}"
        )
        fps_line = (
            f"Runtime : {actual_fps:5.1f} / {target_fps:4.1f} FPS   "
            f"Frame: {state.frame_idx:<7d}  Scroll X: {state.scroll_x:<5d} px"
        )
        ai_str = (ai_label if ai_label else ("ON" if ai_on else "OFF"))[:4]
        mod_line = (
            f"Modules : Audio={'ON ' if audio_on else 'OFF'}  "
            f"AI={ai_str:<4s} "
            f"Overlay={'ON ' if overlay_on else 'OFF'}  "
            f"P1={input_device[:18]}"
        )

        p1_st = "ALIVE" if state.p1.active else "IDLE "
        p2_st = "ALIVE" if state.p2.active else "IDLE "
        p1_line = (
            f"P1 Human: {p1_st} ({state.p1.x:3d},{state.p1.y:3d}) "
            f"L={state.p1.lives} W={state.p1.weapon_name:<11} "
            f"Sc={state.p1.score:06d} [{format_buttons(p1_keys)[:10]}]"
        )
        p2_line = (
            f"P2 AI   : {p2_st} ({state.p2.x:3d},{state.p2.y:3d}) "
            f"L={state.p2.lives} W={state.p2.weapon_name:<11} "
            f"Sc={state.p2.score:06d} [{format_buttons(p2_keys)[:10]}]"
        )

        coop_tag = _coop_tag(state)
        dist_line = (
            f"Co-Op   : dx={state.p1_p2_dx:+6.1f}  dy={state.p1_p2_dy:+6.1f}  "
            f"Dist={state.p1_p2_distance:5.1f} px   Status: [{coop_tag}]"
        )

        near_e = f"{state.nearest_enemy_dist:5.1f}" if state.nearest_enemy_dist is not None else "  -  "
        near_t = f"{state.nearest_threat_dist:5.1f}" if state.nearest_threat_dist is not None else "  -  "
        threat_line = (
            f"Entities: Enemy={len(state.enemies):2d} ({near_e}px)  "
            f"EBul={len(state.enemy_bullets):2d}  "
            f"PBul={len(state.player_bullets):2d}  "
            f"NearThreat={near_t}px"
        )

        obs = state.obstacles_p2
        obs_line = (
            f"P2 Env  : Ground={'YES' if obs.ground_ahead else 'NO ':3s}  "
            f"Pit={'YES' if obs.pit_ahead else 'NO ':3s}  "
            f"Wall={'YES' if obs.wall_ahead else 'NO ':3s}  "
            f"PlatAbove={'YES' if obs.platform_above else 'NO ':3s}  "
            f"Src={state.source}"
        )

        radar_rows = _render_radar(state, width=40, height=7)
        radar_info = [
            "Mini-Radar (256x224)",
            "1 = Player 1 (Human)",
            "2 = Player 2 (AI)",
            "E = Enemy",
            "* = Enemy Bullet",
            ". = Player Bullet",
            "Esc = Quit game",
        ]

        lines = [
            top,
            row(self._c(BOLD + CYAN, title), raw_len=len(title)),
            mid,
            row(self._c(YELLOW, stage_line), raw_len=len(stage_line)),
            row(fps_line),
            row(mod_line),
            mid,
            row(self._c(GREEN if state.p1.active else DIM, p1_line), raw_len=len(p1_line)),
            row(self._c(CYAN if state.p2.active else DIM, p2_line), raw_len=len(p2_line)),
            row(dist_line),
            mid,
            row(self._c(RED if state.enemy_bullets else RESET, threat_line), raw_len=len(threat_line)),
            row(obs_line),
            mid,
        ]

        for r_grid, info in zip(radar_rows, radar_info):
            combined = f"[{r_grid}]  {info}"
            lines.append(row(combined))

        lines.append(bot)
        return "\n".join(lines)

    def update(
        self,
        state: GameState,
        *,
        actual_fps: float = 0.0,
        target_fps: float = 60.0,
        p1_keys: Iterable[int] = (),
        p2_keys: Iterable[int] = (),
        audio_on: bool = True,
        ai_on: bool = True,
        ai_label: str | None = None,
        overlay_on: bool = False,
        input_device: str = "Keyboard",
        force: bool = False,
    ) -> bool:
        if not self.enabled or not self.is_tty:
            return False
        now = time.perf_counter()
        if not force and (now - self._last_draw) < self.refresh_interval:
            return False
        self._last_draw = now
        if not self._started:
            self.stream.write("\x1b[?25l\x1b[2J")
            self._started = True
        text = self.render_text(
            state,
            actual_fps=actual_fps,
            target_fps=target_fps,
            p1_keys=p1_keys,
            p2_keys=p2_keys,
            audio_on=audio_on,
            ai_on=ai_on,
            ai_label=ai_label,
            overlay_on=overlay_on,
            input_device=input_device,
        )
        self.stream.write("\x1b[H" + text + "\n")
        self.stream.flush()
        return True

    def close(self) -> None:
        if self._started and self.is_tty:
            self.stream.write("\x1b[?25h\n")
            self.stream.flush()
            self._started = False
