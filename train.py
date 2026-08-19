"""
REINFORCE training for CarRacing.

The bot watches the track (a 96x96 image), picks a move, and after each full
lap attempt (episode) we look back at how well it did. Good episodes push its
decisions to be more likely next time; bad episodes push them to be less likely.
"""
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from policy import PolicyNet

NUM_EPISODES = 300
GAMMA = 0.99
LEARNING_RATE = 1e-4
FRAME_SKIP = 4          # repeat each chosen move for 4 frames -> faster training, smoother control
MAX_DECISIONS = 250     # ~1000 raw frames per episode, cut into chunks of FRAME_SKIP
ENTROPY_COEF = 0.01     # small bonus for staying a bit exploratory instead of freezing on one move
CHECKPOINT_EVERY = 25
LOG_PATH = "reward_log.json"
CHECKPOINT_PATH = "checkpoint.pt"


def to_tensor(obs):
    # obs is HxWx3 uint8 -> 1x3xHxW float in [0, 1]
    t = torch.from_numpy(obs.copy()).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def run_episode(env, policy, device):
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    entropies = []

    for _ in range(MAX_DECISIONS):
        state = to_tensor(obs).to(device)
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        total_reward = 0.0
        done = False
        for _ in range(FRAME_SKIP):
            obs, r, terminated, truncated, _ = env.step(action.item())
            total_reward += r
            done = terminated or truncated
            if done:
                break
        rewards.append(total_reward)

        if done:
            break

    return log_probs, rewards, entropies


def compute_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        returns[i] = running
    # normalize -> stabilizes training (this is the "baseline" variance-reduction trick)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return torch.from_numpy(returns)


def main():
    device = torch.device("cpu")
    env = gym.make("CarRacing-v3", continuous=False)
    policy = PolicyNet(num_actions=env.action_space.n).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    reward_history = []
    start = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        log_probs, rewards, entropies = run_episode(env, policy, device)
        episode_reward = sum(rewards)
        reward_history.append(episode_reward)

        returns = compute_returns(rewards, GAMMA).to(device)
        log_probs_t = torch.cat(log_probs)
        entropy_t = torch.cat(entropies)

        loss = -(log_probs_t * returns).sum() - ENTROPY_COEF * entropy_t.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_last_20 = np.mean(reward_history[-20:])
        elapsed = time.time() - start
        print(
            f"episode {episode:4d} | reward {episode_reward:8.1f} | "
            f"avg(last20) {avg_last_20:8.1f} | elapsed {elapsed/60:.1f}m",
            flush=True,
        )

        if episode % CHECKPOINT_EVERY == 0:
            torch.save(policy.state_dict(), CHECKPOINT_PATH)
            with open(LOG_PATH, "w") as f:
                json.dump(reward_history, f)

    torch.save(policy.state_dict(), CHECKPOINT_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(reward_history, f)
    env.close()
    print("done")


if __name__ == "__main__":
    main()
