"""Policy wrappers for Contra AI Co-Player (RuleAI & trained PPO)."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import numpy as np

from .state import GameState

# Libretro joypad button IDs (duplicated locally to avoid circular imports)
B = 0


class PPOPolicy:
    """Loads a Stable-Baselines3 PPO model and outputs Libretro joypad button sets for P2.

    Automatically matches the model's frame-stack depth and training frame_skip cadence so
    live 60 FPS inference mirrors ContraCoPlayerEnv dynamics.
    """

    def __init__(
        self,
        model_path: str | Path,
        deterministic: bool = True,
        device: str = "cpu",
        frame_skip: int | None = None,
    ):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required for PPO policy. Install with: pip install -e '.[rl]'"
            ) from exc
        from .env import ACTIONS, OBS_DIM, encode_observation

        resolved = Path(model_path).expanduser().resolve()
        self.model_path = str(resolved)
        self.model = PPO.load(self.model_path, device=device)
        self.deterministic = deterministic
        self._actions = ACTIONS
        self._encode = encode_observation

        # Auto-detect frame_stack from model observation space (backward compatible with 38-dim models)
        obs_dim = int(self.model.observation_space.shape[0])
        self.frame_stack = max(1, obs_dim // OBS_DIM)

        # Auto-detect frame_skip from sibling JSON metadata if not explicitly provided
        detected_skip = 4
        meta_path = resolved.with_suffix(".json")
        if frame_skip is None and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                detected_skip = int(meta.get("frame_skip", 4))
            except (OSError, ValueError, TypeError):
                detected_skip = 4
        self.frame_skip = max(1, int(frame_skip if frame_skip is not None else detected_skip))

        self._obs_stack: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._cached_keys: set[int] = set()
        self._hold_left = 0
        self._t = 0

    def reset(self) -> None:
        self._obs_stack.clear()
        self._cached_keys = set()
        self._hold_left = 0

    def _build_obs(self, state: GameState) -> np.ndarray:
        obs = self._encode(state)
        if self.frame_stack <= 1:
            return obs
        if not self._obs_stack:
            for _ in range(self.frame_stack):
                self._obs_stack.append(obs)
        else:
            self._obs_stack.append(obs)
        return np.concatenate(list(self._obs_stack), axis=0)

    def act(self, frame: np.ndarray | None, state: GameState | None = None) -> set[int]:
        self._t += 1
        if state is None or not state.p2.active:
            self.reset()
            return set()

        if self._hold_left <= 0:
            obs = self._build_obs(state)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self._cached_keys = set(self._actions[int(action)])
            self._hold_left = self.frame_skip

        self._hold_left -= 1
        keys = set(self._cached_keys)
        # Pulse B so NES default rifle fires continuously when model holds shoot
        if B in keys and (self._t % 6 >= 3):
            keys.discard(B)
        return keys
