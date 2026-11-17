"""Fast checks that need no trained model and no GPU."""
import os
import sys

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_episode, summarise  # noqa: E402
from policy import PolicyNet  # noqa: E402
from reward_wrapper import BONUS_PER_DECISION, FRAME_SKIP, CarRacingSpeedBonus  # noqa: E402
from train_gpu import compute_returns, normalise_returns  # noqa: E402


def test_policy_maps_a_frame_to_one_logit_per_action():
    out = PolicyNet(num_actions=5)(torch.zeros(3, 3, 96, 96))
    assert out.shape == (3, 5)


def test_discounted_returns_by_hand():
    rewards = np.array([[1.0], [2.0], [4.0]], dtype=np.float32)
    got = compute_returns(rewards, gamma=0.5)
    assert np.allclose(got[:, 0], [1 + 0.5 * 2 + 0.25 * 4, 2 + 0.5 * 4, 4])


def test_normalised_returns_ignore_steps_after_the_episode_ended():
    # Env 0 plays 4 steps; env 1 ends after 2 and is padding afterwards.
    returns = np.array([[4.0, 10.0], [3.0, 20.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    masks = np.array([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=np.float32)
    got = normalise_returns(returns, masks)

    live = masks > 0
    assert abs(got[live].mean()) < 1e-6
    assert abs(got[live].std() - 1.0) < 1e-5
    assert (got[~live] == 0).all()

    # Padding must not change what the played steps get.
    longer = np.vstack([returns, np.zeros((6, 2), dtype=np.float32)])
    longer_masks = np.vstack([masks, np.zeros((6, 2), dtype=np.float32)])
    assert np.allclose(normalise_returns(longer, longer_masks)[:4], got)


class _FakeTrack(gym.Env):
    """Ends on a chosen frame with a chosen terminal reason; 1 point per frame."""

    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(5)

    def __init__(self, end_frame, lap_finished):
        self.end_frame, self.lap_finished, self.frame = end_frame, lap_finished, 0

    def reset(self, **kwargs):
        self.frame = 0
        return 0, {}

    def step(self, action):
        self.frame += 1
        done = self.frame >= self.end_frame
        info = {"lap_finished": self.lap_finished} if done else {}
        return 0, 1.0, done, False, info


def _play(env):
    env.reset()
    total, terminated, info = 0.0, False, {}
    while not terminated:
        _, r, terminated, _, info = env.step(0)
        total += r
    return total, info


def test_speed_bonus_paid_on_a_finished_lap_only():
    # Lap finished on decision 10 against a par of 15: 5 decisions early.
    total, info = _play(CarRacingSpeedBonus(_FakeTrack(10 * FRAME_SKIP, True), par_decisions=15))
    assert info["speed_bonus"] == 5 * BONUS_PER_DECISION
    assert total == 10 * FRAME_SKIP + 5 * BONUS_PER_DECISION

    # Same length, but the car left the playfield: base reward only.
    total, info = _play(CarRacingSpeedBonus(_FakeTrack(10 * FRAME_SKIP, False), par_decisions=15))
    assert "speed_bonus" not in info
    assert total == 10 * FRAME_SKIP


def test_wrapper_stops_stepping_once_the_episode_is_over():
    inner = _FakeTrack(end_frame=6, lap_finished=False)
    env = CarRacingSpeedBonus(inner, par_decisions=15)
    env.reset()
    env.step(0)
    _, r, terminated, _, _ = env.step(0)
    assert terminated and r == 2.0 and inner.frame == 6


def test_summary_counts_outcomes_and_reports_an_interval():
    eps = [
        {"score": 900.0, "outcome": "lap_finished", "tiles_visited_frac": 1.0},
        {"score": 700.0, "outcome": "time_limit", "tiles_visited_frac": 0.8},
        {"score": -50.0, "outcome": "left_playfield", "tiles_visited_frac": 0.1},
    ]
    s = summarise(eps)
    assert (s["laps_finished"], s["hit_time_limit"], s["left_playfield"]) == (1, 1, 1)
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]
    assert abs(s["frac_score_900_plus"] - 1 / 3) < 1e-9


def test_a_seeded_episode_is_reproducible():
    torch.manual_seed(0)
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    policy = PolicyNet(num_actions=5).eval()
    for mode in ("greedy", "sample"):
        a = run_episode(env, policy, seed=7, mode=mode)
        b = run_episode(env, policy, seed=7, mode=mode)
        assert a == b
        assert a["outcome"] in {"time_limit", "lap_finished", "left_playfield"}
        assert a["frames"] <= 1000
    env.close()
