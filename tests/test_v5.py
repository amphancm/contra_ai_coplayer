from pathlib import Path
import numpy as np
import pytest

from contra_coplayer.env import (
    ACTIONS,
    OBS_DIM,
    ContraCoPlayerEnv,
    RewardConfig,
    compute_reward,
    encode_observation,
)
from contra_coplayer.eval_model import evaluate_policy, format_eval_report
from contra_coplayer.libretro import LibretroHost
from contra_coplayer.policy import PPOPolicy
from contra_coplayer.state import Entity, GameState, ObstacleInfo, PlayerState
from contra_coplayer.train import train_ppo


def _sample_state(
    *,
    scroll_x=100,
    p1_x=140,
    p1_y=100,
    p2_x=120,
    p2_y=100,
    p2_lives=2,
    p1_score=0,
    p2_score=0,
    stage=1,
):
    dist = float(np.hypot(p2_x - p1_x, p2_y - p1_y))
    return GameState(
        frame_idx=10,
        phase_id=4,
        phase="PLAYING",
        stage=stage,
        stage_name="Jungle",
        two_player=True,
        scroll_x=scroll_x,
        p1=PlayerState(1, True, p1_x, p1_y, 2, 0, "Rifle", p1_score),
        p2=PlayerState(2, True, p2_x, p2_y, p2_lives, 0, "Rifle", p2_score),
        enemies=[Entity(0, "enemy", 3, 180, 100)],
        enemy_bullets=[Entity(1, "enemy_bullet", 5, 150, 100)],
        p1_p2_dx=float(p2_x - p1_x),
        p1_p2_dy=float(p2_y - p1_y),
        p1_p2_distance=dist,
        obstacles_p2=ObstacleInfo(ground_ahead=True),
    )


def test_encode_observation_shape_and_bounds():
    st = _sample_state()
    obs = encode_observation(st)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert obs[2] == 1.0   # p2 active
    assert obs[11] == 1.0  # ground ahead
    assert obs[17] == 1.0  # first enemy present
    assert obs[29] == 1.0  # first enemy bullet present


def test_compute_reward_progress_score_coop_and_death():
    cfg = RewardConfig()
    prev = _sample_state(scroll_x=100, p2_score=0, p2_lives=2)
    curr_good = _sample_state(scroll_x=120, p2_score=200, p1_score=100, p2_lives=2)
    r_good, terms_good = compute_reward(prev, curr_good, cfg)
    assert terms_good["progress"] > 0
    assert terms_good["p2_score"] > 0
    assert terms_good["p1_score"] > 0
    assert terms_good["coop"] > 0
    assert terms_good["death"] == 0.0
    assert r_good > 0

    curr_dead = _sample_state(scroll_x=100, p2_lives=1)
    r_dead, terms_dead = compute_reward(prev, curr_dead, cfg)
    assert terms_dead["death"] == cfg.death_penalty
    assert r_dead < 0


def test_libretro_save_and_load_state_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    core = root / "cores" / "fceumm_libretro.so"
    rom = root / "roms" / "Contra (U).nes"
    if not core.is_file() or not rom.is_file():
        pytest.skip("FCEUmm core or Contra ROM not present")

    state_file = tmp_path / "test.state"
    with LibretroHost(core, rom) as host:
        for _ in range(30):
            host.step()
        ram_before = host.read_ram()
        host.save_state_file(state_file)
        assert state_file.is_file() and state_file.stat().st_size > 0
        for _ in range(30):
            host.step()
        host.load_state_file(state_file)
        ram_after = host.read_ram()
        assert ram_after == ram_before


def test_contra_env_and_ppo_train_eval_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    core = root / "cores" / "fceumm_libretro.so"
    rom = root / "roms" / "Contra (U).nes"
    if not core.is_file() or not rom.is_file():
        pytest.skip("FCEUmm core or Contra ROM not present")

    state_path = tmp_path / "stage1_2p.state"
    model_prefix = tmp_path / "ppo_smoke"

    # 1) Quick env reset + step check
    env = ContraCoPlayerEnv(core, rom, state_path=state_path, frame_skip=4, max_steps=20)
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (OBS_DIM,)
        assert state_path.is_file()
        assert info["state"].phase == "PLAYING"
        obs2, reward, terminated, truncated, info2 = env.step(3)  # RIGHT + B
        assert obs2.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert not terminated
    finally:
        env.close()

    # 2) Smoke-train PPO (64 timesteps on CPU for fast unit test)
    saved_zip = train_ppo(
        str(core),
        str(rom),
        out_path=str(model_prefix),
        state_path=str(state_path),
        timesteps=64,
        n_steps=32,
        batch_size=32,
        max_steps=40,
        device="cpu",
        seed=42,
    )
    assert saved_zip.is_file()
    assert saved_zip.with_suffix(".json").is_file()

    # 3) Load PPOPolicy and verify live .act() returns valid button set
    policy = PPOPolicy(saved_zip, device="cpu")
    keys = policy.act(None, _sample_state())
    assert isinstance(keys, set)
    assert keys.issubset(set().union(*ACTIONS))

    # 4) Evaluate both PPO model and RuleAI baseline
    ppo_eval = evaluate_policy(
        str(core), str(rom), model_path=str(saved_zip), state_path=str(state_path), episodes=1, max_steps=30
    )
    base_eval = evaluate_policy(
        str(core), str(rom), model_path=None, state_path=str(state_path), episodes=1, max_steps=30
    )
    assert "Evaluation Report" in format_eval_report(ppo_eval)
    assert base_eval.episodes == 1
