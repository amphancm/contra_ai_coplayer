"""V5/V6 PPO Training Pipeline (supports parallel SubprocVecEnv & resume)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .env import ContraCoPlayerEnv, bootstrap_stage1_2p_state
from .libretro import LibretroHost
from .state import GameStateDetector


def _make_env(
    core_path: str,
    rom_path: str,
    state_path: str | None,
    frame_skip: int,
    frame_stack: int,
    max_steps: int,
    seed: int,
    p1_partner: str = "diverse",
    terminate_on_p2_death: bool = False,
):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = ContraCoPlayerEnv(
            core_path=core_path,
            rom_path=rom_path,
            state_path=state_path,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            max_steps=max_steps,
            p1_partner=p1_partner,
            terminate_on_p2_death=terminate_on_p2_death,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def _linear_lr_schedule(initial_lr: float, min_ratio: float = 0.1, start_progress: float = 1.0):
    span = max(float(start_progress), 1e-9)

    def _schedule(progress_remaining: float) -> float:
        local_progress = min(1.0, max(0.0, progress_remaining / span))
        return initial_lr * (min_ratio + (1.0 - min_ratio) * local_progress)

    return _schedule


def _ensure_bootstrap_state(core_path: str, rom_path: str, state_path: str | None) -> None:
    if not state_path:
        return
    target = Path(state_path)
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with LibretroHost(core_path, rom_path) as host:
        data = bootstrap_stage1_2p_state(host, GameStateDetector())
        target.write_bytes(data)


def train_ppo(
    core_path: str,
    rom_path: str,
    *,
    out_path: str = "models/ppo_contra",
    state_path: str | None = "states/stage1_2p.state",
    timesteps: int = 50_000,
    n_envs: int = 1,
    n_steps: int = 512,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.015,
    net_arch: tuple[int, ...] = (128, 128),
    frame_skip: int = 4,
    frame_stack: int = 4,
    max_steps: int = 1500,
    p1_partner: str = "diverse",
    terminate_on_p2_death: bool = False,
    lr_decay: bool = True,
    device: str = "auto",
    seed: int = 42,
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
    eval_episodes: int = 5,
    resume_path: str | None = None,
) -> Path:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "RL dependencies missing. Install with: pip install -e '.[rl]'"
        ) from exc

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _ensure_bootstrap_state(core_path, rom_path, state_path)

    n_envs = max(1, int(n_envs))
    env_fns = [
        _make_env(
            core_path,
            rom_path,
            state_path,
            frame_skip,
            frame_stack,
            max_steps,
            seed + rank,
            p1_partner=p1_partner,
            terminate_on_p2_death=terminate_on_p2_death,
        )
        for rank in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    callbacks = []
    if checkpoint_freq > 0:
        ckpt_dir = out_file.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq // n_envs),
                save_path=str(ckpt_dir),
                name_prefix=out_file.stem,
            )
        )

    eval_env = None
    best_model_dir: Path | None = None
    if eval_freq > 0:
        best_model_dir = out_file.parent / f"{out_file.stem}_best"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_partner = "rule" if p1_partner == "diverse" else p1_partner
        eval_env = DummyVecEnv([
            _make_env(
                core_path,
                rom_path,
                state_path,
                frame_skip,
                frame_stack,
                min(max_steps, 1000),
                seed + 10_000,
                p1_partner=eval_partner,
                terminate_on_p2_death=terminate_on_p2_death,
            )
        ])
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                eval_freq=max(1, eval_freq // n_envs),
                n_eval_episodes=max(1, int(eval_episodes)),
                deterministic=True,
                render=False,
            )
        )

    t0 = time.perf_counter()
    try:
        if resume_path:
            model = PPO.load(resume_path, env=vec_env, device=device)
            start_prog = timesteps / max(1, int(model.num_timesteps) + int(timesteps))
            resume_lr = _linear_lr_schedule(learning_rate, start_progress=start_prog) if lr_decay else learning_rate
            model.learning_rate = resume_lr
            model._setup_lr_schedule()
            model.ent_coef = ent_coef
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callbacks or None)
        else:
            lr_arg = _linear_lr_schedule(learning_rate) if lr_decay else learning_rate
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=lr_arg,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                policy_kwargs={"net_arch": list(net_arch)},
                seed=seed,
                device=device,
                verbose=1,
            )
            model.learn(total_timesteps=timesteps, callback=callbacks or None)
        model.save(str(out_file))
    finally:
        vec_env.close()
        if eval_env is not None:
            eval_env.close()

    elapsed = time.perf_counter() - t0
    saved_zip = out_file if out_file.suffix == ".zip" else out_file.with_suffix(".zip")
    best_zip = (best_model_dir / "best_model.zip") if best_model_dir else None
    summary = {
        "model_path": str(saved_zip.resolve()),
        "best_model_path": str(best_zip.resolve()) if (best_zip and best_zip.is_file()) else None,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "elapsed_sec": round(elapsed, 2),
        "fps_steps_per_sec": round(timesteps / max(elapsed, 1e-6), 1),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_decay": lr_decay,
        "ent_coef": ent_coef,
        "net_arch": list(net_arch),
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "p1_partner": p1_partner,
        "terminate_on_p2_death": terminate_on_p2_death,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "device": str(model.device),
        "seed": seed,
        "resumed_from": resume_path,
    }
    summary_text = json.dumps(summary, indent=2)
    saved_zip.with_suffix(".json").write_text(summary_text, encoding="utf-8")
    if best_zip and best_zip.is_file():
        best_zip.with_suffix(".json").write_text(summary_text, encoding="utf-8")
    print(f"Saved PPO model to {saved_zip} ({elapsed:.1f}s, {summary['fps_steps_per_sec']} steps/s)")
    if best_zip and best_zip.is_file():
        print(f"Best eval model saved to {best_zip}")
    return saved_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Contra P2 Co-Player using PPO (V5/V6)")
    parser.add_argument("--core", required=True, help="Path to fceumm_libretro.so")
    parser.add_argument("--rom", required=True, help="Path to Contra NES ROM")
    parser.add_argument("--out", default="models/ppo_contra", help="Output model path prefix")
    parser.add_argument("--save-state", default="states/stage1_2p.state", help="Bootstrap save state file")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel environments (SubprocVecEnv when >1)")
    parser.add_argument("--n-steps", type=int, default=512, help="Rollout steps per env per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--no-lr-decay", action="store_true", help="Disable linear learning-rate decay")
    parser.add_argument("--ent-coef", type=float, default=0.015, help="Entropy coefficient")
    parser.add_argument("--net-arch", default="128,128", help="Hidden layer sizes, e.g. 256,256")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frames per RL action")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of stacked observations per step")
    parser.add_argument("--max-steps", type=int, default=1500, help="Max RL steps per episode")
    parser.add_argument(
        "--p1-partner",
        choices=("diverse", "rule", "scripted"),
        default="diverse",
        help="Simulated P1 partner policy (default: diverse mix of RuleP1Partner & ScriptedP1Partner)",
    )
    parser.add_argument("--terminate-on-death", action="store_true", help="Terminate episode immediately on P2 death")
    parser.add_argument("--device", default="auto", help="PyTorch device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=0, help="Checkpoint interval in total steps (0=off)")
    parser.add_argument("--eval-freq", type=int, default=0, help="EvalCallback interval in total steps (0=off)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per EvalCallback evaluation")
    parser.add_argument("--resume", help="Path to existing PPO .zip model to continue training")
    args = parser.parse_args()

    arch = tuple(int(x.strip()) for x in args.net_arch.split(",") if x.strip())
    train_ppo(
        args.core,
        args.rom,
        out_path=args.out,
        state_path=args.save_state,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        net_arch=arch,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        max_steps=args.max_steps,
        p1_partner=args.p1_partner,
        terminate_on_p2_death=args.terminate_on_death,
        lr_decay=not args.no_lr_decay,
        device=args.device,
        seed=args.seed,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
