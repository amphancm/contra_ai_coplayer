"""V5 Gymnasium Environment, Observation Encoder, and Co-Op Reward Function for Contra."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .libretro import LibretroHost
from .state import GameState, GameStateDetector

# Libretro joypad button IDs
B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A = 0, 1, 2, 3, 4, 5, 6, 7, 8

# Discrete action set for P2 AI (12 expressive Contra actions)
ACTIONS: tuple[frozenset[int], ...] = (
    frozenset(),                    # 0: NOOP
    frozenset({RIGHT}),             # 1: Walk right
    frozenset({LEFT}),              # 2: Walk left
    frozenset({RIGHT, B}),          # 3: Walk right + shoot
    frozenset({LEFT, B}),           # 4: Walk left + shoot
    frozenset({RIGHT, A}),          # 5: Jump right
    frozenset({RIGHT, A, B}),       # 6: Jump right + shoot
    frozenset({LEFT, A, B}),        # 7: Jump left + shoot
    frozenset({UP, RIGHT, B}),      # 8: Aim up-right + shoot
    frozenset({DOWN, RIGHT, B}),    # 9: Aim down-right + shoot
    frozenset({DOWN, B}),           # 10: Crouch / prone + shoot
    frozenset({B}),                 # 11: Stand + shoot
)

ACTION_NAMES: tuple[str, ...] = (
    "NOOP", "RIGHT", "LEFT", "RIGHT+B", "LEFT+B",
    "RIGHT+A", "RIGHT+A+B", "LEFT+A+B", "UP+RIGHT+B",
    "DOWN+RIGHT+B", "DOWN+B", "B",
)

OBS_DIM = 38


def encode_observation(state: GameState) -> np.ndarray:
    """Encode V4 GameState into a normalized 38-dim float32 vector in [-1.0, 1.0]."""
    vec = np.zeros(OBS_DIM, dtype=np.float32)

    # 0..5: P2 state
    vec[0] = state.p2.x / 256.0
    vec[1] = state.p2.y / 240.0
    vec[2] = 1.0 if state.p2.active else 0.0
    vec[3] = min(1.0, state.p2.lives / 5.0)
    vec[4] = min(1.0, (state.p2.weapon_id & 0x0F) / 8.0)
    vec[5] = min(1.0, state.scroll_x / 4096.0)

    # 6..10: P1 & Co-Op relative
    vec[6] = state.p1.x / 256.0
    vec[7] = state.p1.y / 240.0
    vec[8] = 1.0 if state.p1.active else 0.0
    vec[9] = np.clip(state.p1_p2_dx / 256.0, -1.0, 1.0)
    vec[10] = np.clip(state.p1_p2_dy / 240.0, -1.0, 1.0)

    # 11..14: P2 obstacles
    obs = state.obstacles_p2
    vec[11] = 1.0 if obs.ground_ahead else 0.0
    vec[12] = 1.0 if obs.pit_ahead else 0.0
    vec[13] = 1.0 if obs.wall_ahead else 0.0
    vec[14] = 1.0 if obs.platform_above else 0.0

    # 15..26: Nearest 4 enemies relative to P2 (dx, dy, present)
    enemies_sorted = sorted(
        state.enemies,
        key=lambda e: math.hypot(e.x - state.p2.x, e.y - state.p2.y),
    )[:4]
    for i, e in enumerate(enemies_sorted):
        base = 15 + i * 3
        vec[base] = np.clip((e.x - state.p2.x) / 256.0, -1.0, 1.0)
        vec[base + 1] = np.clip((e.y - state.p2.y) / 240.0, -1.0, 1.0)
        vec[base + 2] = 1.0

    # 27..35: Nearest 3 enemy bullets relative to P2 (dx, dy, present)
    bullets_sorted = sorted(
        state.enemy_bullets,
        key=lambda b: math.hypot(b.x - state.p2.x, b.y - state.p2.y),
    )[:3]
    for i, b in enumerate(bullets_sorted):
        base = 27 + i * 3
        vec[base] = np.clip((b.x - state.p2.x) / 256.0, -1.0, 1.0)
        vec[base + 1] = np.clip((b.y - state.p2.y) / 240.0, -1.0, 1.0)
        vec[base + 2] = 1.0

    # 36..37: Entity density
    vec[36] = min(1.0, len(state.enemies) / 16.0)
    vec[37] = min(1.0, len(state.enemy_bullets) / 8.0)

    return np.clip(vec, -1.0, 1.0)


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 0.04
    p2_score_weight: float = 0.02
    p1_score_weight: float = 0.005
    survival_bonus: float = 0.01
    coop_sync_bonus: float = 0.02
    coop_sync_max_dist: float = 85.0
    too_far_penalty: float = -0.03
    too_far_dist: float = 115.0
    weapon_upgrade_bonus: float = 3.0
    danger_penalty: float = -0.04
    danger_dist: float = 18.0
    death_penalty: float = -6.0
    stage_clear_bonus: float = 25.0


def compute_reward(
    prev: GameState,
    curr: GameState,
    cfg: RewardConfig = RewardConfig(),
) -> tuple[float, dict[str, float]]:
    """Compute shaped Co-Op reward for Player 2 from consecutive GameStates."""
    terms: dict[str, float] = {}

    delta_scroll = max(0, curr.scroll_x - prev.scroll_x)
    terms["progress"] = delta_scroll * cfg.progress_weight

    delta_p2_score = max(0, curr.p2.score - prev.p2.score)
    terms["p2_score"] = delta_p2_score * cfg.p2_score_weight

    delta_p1_score = max(0, curr.p1.score - prev.p1.score)
    terms["p1_score"] = delta_p1_score * cfg.p1_score_weight

    terms["survival"] = cfg.survival_bonus if curr.p2.active else 0.0

    coop = 0.0
    if curr.p1.active and curr.p2.active:
        if curr.p1_p2_distance <= cfg.coop_sync_max_dist:
            coop = cfg.coop_sync_bonus
        elif curr.p1_p2_distance >= cfg.too_far_dist:
            coop = cfg.too_far_penalty
    terms["coop"] = coop

    weapon_up = curr.p2.active and (curr.p2.weapon_id > 0) and (curr.p2.weapon_id != prev.p2.weapon_id)
    terms["weapon"] = cfg.weapon_upgrade_bonus if weapon_up else 0.0

    in_danger = (
        curr.p2.active
        and curr.nearest_threat_dist is not None
        and curr.nearest_threat_dist < cfg.danger_dist
    )
    terms["danger"] = cfg.danger_penalty if in_danger else 0.0

    p2_died = curr.p2.lives < prev.p2.lives
    terms["death"] = cfg.death_penalty if p2_died else 0.0

    stage_clear = curr.stage > prev.stage
    terms["stage_clear"] = cfg.stage_clear_bonus if stage_clear else 0.0

    total = float(sum(terms.values()))
    return total, terms


class ScriptedP1Partner:
    """Simulated human-like Player 1 partner for offline RL training."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.t = 0
        self._pause_until = 0
        self._jump_until = 0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.t = 0
        self._pause_until = 0
        self._jump_until = 0

    def act(self, state: GameState | None) -> set[int]:
        self.t += 1
        keys: set[int] = set()
        if self.t % 6 < 3:
            keys.add(B)
        if self.t >= self._pause_until:
            if self.rng.random() < 0.015:
                self._pause_until = self.t + int(self.rng.integers(10, 35))
            else:
                keys.add(RIGHT)
        if self.t < self._jump_until:
            keys.add(A)
        elif self.rng.random() < 0.025:
            self._jump_until = self.t + int(self.rng.integers(10, 18))
            keys.add(A)
        return keys


def bootstrap_stage1_2p_state(host: LibretroHost, detector: GameStateDetector) -> bytes:
    """Boot Contra (U) from power-on into Stage 1 2-Player gameplay and return serialized state."""
    for i in range(900):
        p1: set[int] = set()
        if i in (60, 61):
            p1.add(START)
        elif i in (80, 81):
            p1.add(SELECT)
        elif i in (100, 101):
            p1.add(START)
        frame, _ = host.step(p1, set())
        if i >= 650:
            st = detector.detect(frame, host.read_ram(), frame_idx=i)
            if st.phase == "PLAYING" and st.p1.active and st.p2.active:
                return host.save_state()
    return host.save_state()


class ContraCoPlayerEnv(gym.Env):
    """Gymnasium Environment for training Contra Player 2 AI Co-Player via PPO."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        core_path: str | Path,
        rom_path: str | Path,
        *,
        state_path: str | Path | None = None,
        frame_skip: int = 4,
        max_steps: int = 1500,
        reward_config: RewardConfig = RewardConfig(),
        p1_policy: Callable[[GameState | None], set[int]] | None = None,
        terminate_on_p2_death: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.core_path = str(core_path)
        self.rom_path = str(rom_path)
        self.state_path = Path(state_path) if state_path else None
        self.frame_skip = max(1, int(frame_skip))
        self.max_steps = int(max_steps)
        self.reward_config = reward_config
        self.terminate_on_p2_death = terminate_on_p2_death
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.host = LibretroHost(self.core_path, self.rom_path)
        self.detector = GameStateDetector()
        self._scripted_p1 = ScriptedP1Partner()
        self.p1_policy = p1_policy
        self._init_state: bytes | None = None
        self._last_frame: np.ndarray | None = None
        self._last_state: GameState | None = None
        self._steps = 0
        self._total_frames = 0
        self._ep_deaths = 0
        self._ep_dist_sum = 0.0
        self._ep_dist_count = 0

    def _ensure_init_state(self) -> bytes:
        if self._init_state is not None:
            return self._init_state
        if self.state_path and self.state_path.is_file():
            self._init_state = self.state_path.read_bytes()
            return self._init_state
        self._init_state = bootstrap_stage1_2p_state(self.host, self.detector)
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_bytes(self._init_state)
        return self._init_state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._scripted_p1.reset(self.np_random)
        init_bytes = self._ensure_init_state()
        self.host.load_state(init_bytes)
        frame, _ = self.host.step(set(), set())
        self._last_frame = frame
        self._steps = 0
        self._total_frames = 0
        self._ep_deaths = 0
        self._ep_dist_sum = 0.0
        self._ep_dist_count = 0
        self._last_state = self.detector.detect(frame, self.host.read_ram(), frame_idx=0)
        return encode_observation(self._last_state), {"state": self._last_state}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._last_state is not None, "Call reset() before step()"
        prev_state = self._last_state
        p2_keys = set(ACTIONS[int(action)])

        frame = self._last_frame
        for _ in range(self.frame_skip):
            p1_keys = self.p1_policy(self._last_state) if self.p1_policy else self._scripted_p1.act(self._last_state)
            # Toggle fire slightly on held B so NES semi-auto rifle fires continuously
            step_p2 = set(p2_keys)
            if B in step_p2 and (self._total_frames % 6 >= 3):
                step_p2.discard(B)
            frame, _ = self.host.step(p1_keys, step_p2)
            self._total_frames += 1

        self._last_frame = frame
        curr_state = self.detector.detect(frame, self.host.read_ram(), frame_idx=self._total_frames)
        self._last_state = curr_state
        self._steps += 1

        reward, terms = compute_reward(prev_state, curr_state, self.reward_config)
        if curr_state.p2.lives < prev_state.p2.lives:
            self._ep_deaths += 1
        if curr_state.p1.active and curr_state.p2.active:
            self._ep_dist_sum += curr_state.p1_p2_distance
            self._ep_dist_count += 1

        terminated = False
        if curr_state.phase == "GAME_OVER":
            terminated = True
        elif self.terminate_on_p2_death and curr_state.p2.lives < prev_state.p2.lives:
            terminated = True
        elif curr_state.p2.lives == 0 and not curr_state.p2.active and self._ep_deaths > 0:
            terminated = True

        truncated = self._steps >= self.max_steps
        info = {
            "state": curr_state,
            "reward_terms": terms,
            "scroll_x": curr_state.scroll_x,
            "p2_score": curr_state.p2.score,
            "p1_score": curr_state.p1.score,
            "p2_deaths": self._ep_deaths,
            "mean_p1_p2_dist": (self._ep_dist_sum / self._ep_dist_count) if self._ep_dist_count else 0.0,
        }
        return encode_observation(curr_state), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._last_frame
        return None

    def close(self) -> None:
        self.host.close()
