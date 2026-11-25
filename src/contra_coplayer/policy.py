"""Policy wrappers for Contra AI Co-Player (RuleAI & trained PPO)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .state import GameState

# Libretro joypad button IDs (duplicated locally to avoid circular imports)
B = 0


class PPOPolicy:
    """Loads a Stable-Baselines3 PPO model and outputs Libretro joypad button sets for P2."""

    def __init__(self, model_path: str | Path, deterministic: bool = True, device: str = "auto"):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required for PPO policy. Install with: pip install -e '.[rl]'"
            ) from exc
        from .env import ACTIONS, encode_observation

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.model = PPO.load(self.model_path, device=device)
        self.deterministic = deterministic
        self._actions = ACTIONS
        self._encode = encode_observation
        self._t = 0

    def act(self, frame: np.ndarray | None, state: GameState | None = None) -> set[int]:
        self._t += 1
        if state is None or not state.p2.active:
            return set()
        obs = self._encode(state)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        keys = set(self._actions[int(action)])
        # Pulse B so NES default rifle fires continuously when model holds shoot
        if B in keys and (self._t % 6 >= 3):
            keys.discard(B)
        return keys
