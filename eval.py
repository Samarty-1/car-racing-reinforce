"""Headless, reproducible evaluation of a trained checkpoint.

    python eval.py                                  # 100 seeded tracks, greedy policy
    python eval.py --mode sample                    # same tracks, actions sampled as in training
    python eval.py --episodes 10 --checkpoint checkpoint.pt

Every episode uses the standard CarRacing-v3 rules: the environment itself ends
an episode after 1000 frames (250 decisions at FRAME_SKIP=4), when the lap is
finished, or when the car leaves the playfield. Track k is always generated
from seed (--seed + k), so two checkpoints or two action modes are compared on
exactly the same tracks.
"""
import argparse
import json

import numpy as np
import torch
import gymnasium as gym

from policy import PolicyNet

FRAME_SKIP = 4  # must match training: one decision is held for 4 frames


def to_tensor(obs):
    t = torch.from_numpy(obs.copy()).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def run_episode(env, policy, seed, mode="greedy"):
    """Play one seeded episode to the environment's own end. Returns a dict."""
    obs, _ = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    total_reward, decisions, frames = 0.0, 0, 0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        with torch.no_grad():
            logits = policy(to_tensor(obs))
        if mode == "greedy":
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1, generator=gen).item())
        for _ in range(FRAME_SKIP):
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += float(r)
            frames += 1
            if terminated or truncated:
                break
        decisions += 1

    core = env.unwrapped
    if terminated:
        # The env reports which of its two terminal conditions fired.
        outcome = "lap_finished" if info.get("lap_finished") else "left_playfield"
    else:
        outcome = "time_limit"
    return {
        "seed": seed,
        "score": total_reward,
        "frames": frames,
        "decisions": decisions,
        "outcome": outcome,
        "tiles_visited_frac": core.tile_visited_count / len(core.track),
    }


def summarise(episodes):
    scores = np.array([e["score"] for e in episodes], dtype=float)
    n = len(scores)
    sem = float(scores.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    outcomes = [e["outcome"] for e in episodes]
    return {
        "episodes": n,
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if n > 1 else float("nan"),
        "ci95_low": float(scores.mean() - 1.96 * sem),
        "ci95_high": float(scores.mean() + 1.96 * sem),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "frac_score_900_plus": float((scores >= 900).mean()),
        "laps_finished": outcomes.count("lap_finished"),
        "hit_time_limit": outcomes.count("time_limit"),
        "left_playfield": outcomes.count("left_playfield"),
        "mean_tiles_visited_frac": float(np.mean([e["tiles_visited_frac"] for e in episodes])),
    }


def evaluate(checkpoint, episodes=100, seed=0, mode="greedy", verbose=True):
    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    policy = PolicyNet(num_actions=env.action_space.n)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()

    results = []
    for k in range(episodes):
        ep = run_episode(env, policy, seed + k, mode)
        results.append(ep)
        if verbose:
            print(
                f"track seed {ep['seed']:4d}: score {ep['score']:7.1f} | "
                f"{ep['frames']:4d} frames | {ep['outcome']}",
                flush=True,
            )
    env.close()
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "first_seed": seed,
        "frame_skip": FRAME_SKIP,
        "summary": summarise(results),
        "per_episode": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoint_gpu.pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="seed of the first track; track k uses seed+k")
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--out", default=None, help="default: eval_<mode>.json")
    args = p.parse_args()

    # The network is tiny; extra threads only fight the physics engine for cores.
    torch.set_num_threads(1)
    result = evaluate(args.checkpoint, args.episodes, args.seed, args.mode)
    out = args.out or f"eval_{args.mode}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(
        f"\n{args.mode}: mean {s['mean']:.1f} (95% CI {s['ci95_low']:.1f} to {s['ci95_high']:.1f}) | "
        f"min {s['min']:.1f} | max {s['max']:.1f} | laps finished {s['laps_finished']}/{s['episodes']} | "
        f"time limit {s['hit_time_limit']} | left playfield {s['left_playfield']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
