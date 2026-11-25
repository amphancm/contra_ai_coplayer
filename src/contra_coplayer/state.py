"""V4 Game State Detection: direct NES RAM + RGB frame analysis for Contra (U)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

STAGE_NAMES = {
    1: "Jungle",
    2: "Base 1",
    3: "Waterfall",
    4: "Base 2",
    5: "Snow Field",
    6: "Energy Zone",
    7: "Hangar",
    8: "Alien's Lair",
}

PHASE_NAMES = {
    0: "TITLE",
    1: "DEMO",
    2: "STAGE_INTRO",
    3: "STARTING",
    4: "PLAYING",
    5: "STAGE_CLEAR",
    6: "GAME_OVER",
}

WEAPON_NAMES = {
    0: "Rifle",
    1: "Machine Gun",
    2: "Fireball",
    3: "Laser",
    4: "Spread",
}

# Contra (U) enemy object types that represent projectiles/bullets in the 16-slot enemy table.
ENEMY_BULLET_TYPES = frozenset({0x05, 0x06, 0x16, 0x1B, 0x22, 0x27})


@dataclass(frozen=True)
class Entity:
    slot: int
    kind: str  # "enemy", "enemy_bullet", "p1_bullet", "p2_bullet"
    type_id: int
    x: int
    y: int


@dataclass(frozen=True)
class PlayerState:
    port: int  # 1 or 2
    active: bool
    x: int
    y: int
    lives: int
    weapon_id: int
    weapon_name: str
    score: int


@dataclass(frozen=True)
class ObstacleInfo:
    ground_ahead: bool = False
    pit_ahead: bool = False
    wall_ahead: bool = False
    platform_above: bool = False
    ground_y: int | None = None


@dataclass
class GameState:
    frame_idx: int
    phase_id: int
    phase: str
    stage: int
    stage_name: str
    two_player: bool
    scroll_x: int
    p1: PlayerState
    p2: PlayerState
    enemies: list[Entity] = field(default_factory=list)
    enemy_bullets: list[Entity] = field(default_factory=list)
    player_bullets: list[Entity] = field(default_factory=list)
    p1_p2_dx: float = 0.0
    p1_p2_dy: float = 0.0
    p1_p2_distance: float = 0.0
    nearest_enemy_dist: float | None = None
    nearest_threat_dist: float | None = None
    obstacles_p2: ObstacleInfo = field(default_factory=ObstacleInfo)
    source: str = "ram+frame"


def _weapon_name(raw_weapon: int) -> str:
    base = WEAPON_NAMES.get(raw_weapon & 0x0F, f"W{raw_weapon & 0x0F}")
    if raw_weapon & 0x10:
        return f"{base}+R"
    return base


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def analyze_obstacles(frame: np.ndarray | None, x: int, y: int) -> ObstacleInfo:
    """Estimate ground, pit, wall, and platform cues around (x, y) directly from the RGB frame.

    Uses lightweight vectorized luminance/gradient heuristics on the 224x256 RGB frame:
    - Ground/platforms have strong horizontal texture/edges below the player's feet.
    - Pits/water/open sky ahead-below have low edge energy or dark/uniform pixels.
    """
    if frame is None or frame.ndim != 3 or frame.shape[0] < 32 or frame.shape[1] < 32:
        return ObstacleInfo()

    h, w, _ = frame.shape
    cx = max(0, min(w - 1, int(x)))
    cy = max(0, min(h - 1, int(y)))

    # Convert only needed bands to fast float luminance
    gray = (
        frame[:, :, 0].astype(np.float32) * 0.299
        + frame[:, :, 1].astype(np.float32) * 0.587
        + frame[:, :, 2].astype(np.float32) * 0.114
    )

    # 1) Ground directly beneath & slightly ahead of player (feet ~ y+16..y+28, x+6..x+26)
    gx0, gx1 = min(w - 1, cx + 6), min(w, cx + 26)
    gy0, gy1 = min(h - 1, cy + 14), min(h, cy + 32)
    ground_ahead = False
    pit_ahead = False
    ground_y: int | None = None
    if gx1 > gx0 + 2 and gy1 > gy0 + 2:
        patch = gray[gy0:gy1, gx0:gx1]
        v_edges = np.abs(np.diff(patch, axis=0))
        row_energy = v_edges.mean(axis=1)
        max_idx = int(np.argmax(row_energy))
        max_energy = float(row_energy[max_idx])
        bright_ratio = float((patch > 35.0).mean())
        ground_ahead = max_energy > 14.0 and bright_ratio > 0.20
        if ground_ahead:
            ground_y = gy0 + max_idx
        else:
            # Pit/drop if region ahead-below is mostly dark or uniform
            pit_ahead = cy < h - 36 and (bright_ratio < 0.18 or max_energy < 8.0)

    # 2) Wall/obstacle directly ahead at torso height (y-8..y+10, x+10..x+24)
    wx0, wx1 = min(w - 1, cx + 10), min(w, cx + 24)
    wy0, wy1 = max(0, cy - 8), min(h, cy + 10)
    wall_ahead = False
    if wx1 > wx0 + 2 and wy1 > wy0 + 2:
        wpatch = gray[wy0:wy1, wx0:wx1]
        h_edge = float(np.abs(np.diff(wpatch, axis=1)).mean())
        wall_ahead = h_edge > 22.0 and float((wpatch > 40.0).mean()) > 0.35

    # 3) Platform above player (y-48..y-16, x-8..x+16)
    px0, px1 = max(0, cx - 8), min(w, cx + 16)
    py0, py1 = max(0, cy - 48), max(0, cy - 16)
    platform_above = False
    if px1 > px0 + 2 and py1 > py0 + 2:
        ppatch = gray[py0:py1, px0:px1]
        p_edge = float(np.abs(np.diff(ppatch, axis=0)).max(initial=0.0))
        platform_above = p_edge > 45.0 and float((ppatch > 35.0).mean()) > 0.15

    return ObstacleInfo(
        ground_ahead=ground_ahead,
        pit_ahead=pit_ahead,
        wall_ahead=wall_ahead,
        platform_above=platform_above,
        ground_y=ground_y,
    )


class GameStateDetector:
    """Extracts structured Contra game state from NES System RAM and RGB frames."""

    def detect(self, frame: np.ndarray | None, ram: bytes | None, frame_idx: int = 0) -> GameState:
        if not ram or len(ram) < 0x800:
            return self._empty_state(frame_idx, source="frame-only")

        phase_id = ram[0x2C]
        if phase_id == 0xFF:
            return self._empty_state(frame_idx, source="ram+frame" if frame is not None else "ram", phase="BOOT")
        phase = PHASE_NAMES.get(phase_id, f"PHASE_{phase_id}")
        raw_stage = ram[0x30]
        stage = (raw_stage + 1) if raw_stage < 8 else 1
        stage_name = STAGE_NAMES.get(stage, f"Stage {stage}")
        two_player = bool(ram[0x22] == 1 or (ram[0x33] not in (0, 0xFF)) or (ram[0x91] not in (0, 0xFF)))
        scroll_x = ((ram[0x64] << 8) | ram[0x65]) if phase_id >= 2 else 0

        raw_p1_sc = ram[0x7E2] | (ram[0x7E3] << 8)
        raw_p2_sc = ram[0x7E4] | (ram[0x7E5] << 8)
        p1_score = 0 if (raw_p1_sc == 0xFFFF or phase_id < 2) else raw_p1_sc * 100
        p2_score = 0 if (raw_p2_sc == 0xFFFF or phase_id < 2) else raw_p2_sc * 100

        playing = phase_id == 4
        p1_lives = 0 if ram[0x32] == 0xFF else ram[0x32]
        p2_lives = 0 if ram[0x33] == 0xFF else ram[0x33]
        p1 = PlayerState(
            port=1,
            active=playing and ram[0x90] > 0,
            x=ram[0x334] if playing else 0,
            y=ram[0x31A] if playing else 0,
            lives=p1_lives,
            weapon_id=ram[0xAA],
            weapon_name=_weapon_name(ram[0xAA]),
            score=p1_score,
        )
        p2 = PlayerState(
            port=2,
            active=playing and ram[0x91] > 0,
            x=ram[0x335] if playing else 0,
            y=ram[0x31B] if playing else 0,
            lives=p2_lives,
            weapon_id=ram[0xAB],
            weapon_name=_weapon_name(ram[0xAB]),
            score=p2_score,
        )

        enemies: list[Entity] = []
        enemy_bullets: list[Entity] = []
        player_bullets: list[Entity] = []

        if playing:
            # 16 enemy/projectile slots
            for slot in range(16):
                etype = ram[0x4B8 + slot]
                if etype == 0:
                    continue
                ex = ram[0x33E + slot]
                ey = ram[0x324 + slot]
                if ey == 0 or ey >= 240:
                    continue
                if etype in ENEMY_BULLET_TYPES:
                    enemy_bullets.append(Entity(slot=slot, kind="enemy_bullet", type_id=etype, x=ex, y=ey))
                else:
                    enemies.append(Entity(slot=slot, kind="enemy", type_id=etype, x=ex, y=ey))

            # 16 player bullet slots: 0..7 = P1, 8..15 = P2
            for slot in range(16):
                btype = ram[0x388 + slot]
                if btype == 0:
                    continue
                bx = ram[0x3C8 + slot]
                by = ram[0x3B8 + slot]
                if by == 0 or by >= 240:
                    continue
                kind = "p1_bullet" if slot < 8 else "p2_bullet"
                player_bullets.append(Entity(slot=slot, kind=kind, type_id=btype, x=bx, y=by))

        dx = float(p2.x - p1.x) if (p1.active and p2.active) else 0.0
        dy = float(p2.y - p1.y) if (p1.active and p2.active) else 0.0
        dist = math.hypot(dx, dy) if (p1.active and p2.active) else 0.0

        nearest_enemy: float | None = None
        nearest_threat: float | None = None
        if p2.active:
            if enemies:
                nearest_enemy = min(_distance(p2.x, p2.y, e.x, e.y) for e in enemies)
            threats = enemies + enemy_bullets
            if threats:
                nearest_threat = min(_distance(p2.x, p2.y, t.x, t.y) for t in threats)

        obstacles_p2 = analyze_obstacles(frame, p2.x, p2.y) if p2.active else ObstacleInfo()

        return GameState(
            frame_idx=frame_idx,
            phase_id=phase_id,
            phase=phase,
            stage=stage,
            stage_name=stage_name,
            two_player=two_player,
            scroll_x=scroll_x,
            p1=p1,
            p2=p2,
            enemies=enemies,
            enemy_bullets=enemy_bullets,
            player_bullets=player_bullets,
            p1_p2_dx=dx,
            p1_p2_dy=dy,
            p1_p2_distance=dist,
            nearest_enemy_dist=nearest_enemy,
            nearest_threat_dist=nearest_threat,
            obstacles_p2=obstacles_p2,
            source="ram+frame" if frame is not None else "ram",
        )

    def _empty_state(self, frame_idx: int, source: str, phase: str = "UNKNOWN") -> GameState:
        p1 = PlayerState(1, False, 0, 0, 0, 0, "Rifle", 0)
        p2 = PlayerState(2, False, 0, 0, 0, 0, "Rifle", 0)
        return GameState(
            frame_idx=frame_idx,
            phase_id=-1,
            phase=phase,
            stage=1,
            stage_name="Jungle",
            two_player=False,
            scroll_x=0,
            p1=p1,
            p2=p2,
            source=source,
        )
