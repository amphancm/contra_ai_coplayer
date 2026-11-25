"""V5 Model Evaluation CLI: compare trained PPO policy vs RuleAI baseline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np

from .app import RuleAI
from .env import ACTIONS, ContraCoPlayerEnv


@dataclass
class EvalSummary:
    label: str
    episodes: int
    mean_reward: float
    std_reward: float
    mean_scroll_x: float
    mean_p2_score: float
    mean_steps: float
    mean_deaths: float
    mean_p1_p2_dist: float


def _keys_to_action_idx(keys: set[int]) -> int:
    frozen = frozenset(keys)
    for idx, act_keys in enumerate(ACTIONS):
        if act_keys == frozen:
            return idx
    # Closest fallback: prioritize RIGHT+A+B, RIGHT+B, RIGHT
    if 7 in keys and 8 in keys and 0 in keys:
        return 6
    if 7 in keys and 0 in keys:
        return 3
    if 7 in keys:
        return 1
    return 0


def evaluate_policy(
    core_path: str,
    rom_path: str,
    *,
    model_path: str | None = None,
    state_path: str | None = "states/stage1_2p.state",
    episodes: int = 5,
    max_steps: int = 800,
    frame_skip: int = 4,
    deterministic: bool = True,
    seed: int = 123,
) -> EvalSummary:
    ppo_model = None
    label = "RuleAI Baseline"
    if model_path:
        from stable_baselines3 import PPO

        ppo_model = PPO.load(model_path)
        label = f"PPO ({model_path})"

    env = ContraCoPlayerEnv(
        core_path=core_path,
        rom_path=rom_path,
        state_path=state_path,
        frame_skip=frame_skip,
        max_steps=max_steps,
    )

    rewards, scrolls, scores, steps_list, deaths_list, dists = [], [], [], [], [], []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            rule_ai = RuleAI()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            last_info = info
            while not done:
                if ppo_model is not None:
                    action, _ = ppo_model.predict(obs, deterministic=deterministic)
                    action = int(action)
                else:
                    keys = rule_ai.act(None, last_info.get("state"))
                    action = _keys_to_action_idx(keys)
                obs, reward, terminated, truncated, last_info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated
            rewards.append(ep_reward)
            scrolls.append(last_info.get("scroll_x", 0))
            scores.append(last_info.get("p2_score", 0))
            steps_list.append(ep_steps)
            deaths_list.append(last_info.get("p2_deaths", 0))
            dists.append(last_info.get("mean_p1_p2_dist", 0.0))
    finally:
        env.close()

    return EvalSummary(
        label=label,
        episodes=episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_scroll_x=float(np.mean(scrolls)),
        mean_p2_score=float(np.mean(scores)),
        mean_steps=float(np.mean(steps_list)),
        mean_deaths=float(np.mean(deaths_list)),
        mean_p1_p2_dist=float(np.mean(dists)),
    )


def format_eval_report(summary: EvalSummary) -> str:
    lines = [
        "=" * 64,
        f"Evaluation Report: {summary.label}",
        "-" * 64,
        f"Episodes        : {summary.episodes}",
        f"Mean Reward     : {summary.mean_reward:8.2f} +/- {summary.std_reward:.2f}",
        f"Mean Scroll X   : {summary.mean_scroll_x:8.1f} px",
        f"Mean P2 Score   : {summary.mean_p2_score:8.1f}",
        f"Mean Steps      : {summary.mean_steps:8.1f}",
        f"Mean P2 Deaths  : {summary.mean_deaths:8.2f}",
        f"Mean P1-P2 Dist : {summary.mean_p1_p2_dist:8.1f} px",
        "=" * 64,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Contra P2 AI policy (PPO or RuleAI baseline)")
    parser.add_argument("--core", required=True, help="Path to fceumm_libretro.so")
    parser.add_argument("--rom", required=True, help="Path to Contra NES ROM")
    parser.add_argument("--model", help="Path to trained PPO .zip model (omit for RuleAI baseline)")
    parser.add_argument("--save-state", default="states/stage1_2p.state")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic PPO actions")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    summary = evaluate_policy(
        args.core,
        args.rom,
        model_path=args.model,
        state_path=args.save_state,
        episodes=args.episodes,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    print(format_eval_report(summary))


if __name__ == "__main__":
    main()
