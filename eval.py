"""Headless evaluation: run N episodes with the trained bot, no display, just scores."""
import sys
import json
import numpy as np
import torch
import gymnasium as gym

from policy import PolicyNet
from train_gpu import FRAME_SKIP, MAX_DECISIONS

CHECKPOINT_PATH = "checkpoint_gpu.pt"


def to_tensor(obs):
    t = torch.from_numpy(obs.copy()).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_path = sys.argv[2] if len(sys.argv) > 2 else "eval_results.json"
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else CHECKPOINT_PATH

    env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
    policy = PolicyNet(num_actions=env.action_space.n)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()

    scores = []
    steps_list = []
    completed_list = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_DECISIONS:
            with torch.no_grad():
                logits = policy(to_tensor(obs))
                action = torch.argmax(logits, dim=-1).item()
            for _ in range(FRAME_SKIP):
                obs, r, terminated, truncated, _ = env.step(action)
                total_reward += r
                done = terminated or truncated
                if done:
                    break
            steps += 1

        core = env.unwrapped
        completed = bool(terminated and core.tile_visited_count >= len(core.track))
        scores.append(total_reward)
        steps_list.append(steps)
        completed_list.append(completed)
        print(f"episode {ep}: total reward {total_reward:.1f} | steps {steps} | completed {completed}", flush=True)

    env.close()
    scores = np.array(scores)
    completed_arr = np.array(completed_list)
    completed_steps = [s for s, c in zip(steps_list, completed_list) if c]
    result = {
        "scores": scores.tolist(),
        "steps": steps_list,
        "completed": completed_list,
        "mean": float(scores.mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "num_positive": int((scores > 0).sum()),
        "num_completed": int(completed_arr.sum()),
        "avg_steps_when_completed": float(np.mean(completed_steps)) if completed_steps else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nmean {result['mean']:.1f} | min {result['min']:.1f} | max {result['max']:.1f} | "
          f"{result['num_positive']}/{episodes} positive | {result['num_completed']}/{episodes} completed", flush=True)


if __name__ == "__main__":
    main()
