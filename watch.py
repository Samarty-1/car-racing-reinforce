"""Pop open a window and watch the trained bot drive."""
import sys
import torch
import gymnasium as gym

from policy import PolicyNet
from train import to_tensor, FRAME_SKIP

CHECKPOINT_PATH = "checkpoint.pt"


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else CHECKPOINT_PATH

    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    policy = PolicyNet(num_actions=env.action_space.n)
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    policy.eval()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                logits = policy(to_tensor(obs))
                action = torch.argmax(logits, dim=-1).item()
            for _ in range(FRAME_SKIP):
                obs, r, terminated, truncated, _ = env.step(action)
                total_reward += r
                done = terminated or truncated
                if done:
                    break
        print(f"episode {ep}: total reward {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
