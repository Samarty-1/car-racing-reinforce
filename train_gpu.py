"""
REINFORCE for CarRacing, parallelized across many game copies + trained on GPU.

Instead of one car practicing at a time, we run NUM_ENVS copies of the track
at once. Every decision step, all of them hand their current view to the
policy network in a single batch, which runs on the GPU. This gives the GPU
enough work at once to actually be worth using, and lets us gather far more
practice per minute than the single-car version.
"""
import os
import time
import json
import numpy as np
import torch
import gymnasium as gym

from policy import PolicyNet

NUM_ENVS = 10           # game copies running at once
NUM_UPDATES = 1200      # one "update" = one batch rollout across all NUM_ENVS
RESUME = True           # continue from an existing checkpoint/log instead of starting fresh
GAMMA = 0.99
LEARNING_RATE = 1e-4
FRAME_SKIP = 4
MAX_DECISIONS = 250
ENTROPY_COEF = 0.01
CHECKPOINT_EVERY = 10
LOG_PATH = "reward_log_gpu.json"
CHECKPOINT_PATH = "checkpoint_gpu.pt"


def make_env():
    return gym.make("CarRacing-v3", continuous=False)


def to_tensor_batch(obs_batch, device):
    # obs_batch: (N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]
    t = torch.from_numpy(obs_batch.copy()).float() / 255.0
    return t.permute(0, 3, 1, 2).to(device)


def run_rollout(envs, policy, device, num_envs):
    obs, _ = envs.reset()
    active = np.ones(num_envs, dtype=bool)

    log_probs, entropies, rewards, masks = [], [], [], []

    for _ in range(MAX_DECISIONS):
        masks.append(active.copy())

        state = to_tensor_batch(obs, device)
        logits = policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        action_np = action.cpu().numpy()
        still_active = active.copy()
        total_r = np.zeros(num_envs, dtype=np.float32)

        for _ in range(FRAME_SKIP):
            obs, r, terminated, truncated, _ = envs.step(action_np)
            done_now = terminated | truncated
            total_r += r * still_active
            still_active = still_active & (~done_now)
            if not still_active.any():
                break

        rewards.append(total_r)
        active = still_active

        if not active.any():
            break

    return log_probs, entropies, rewards, masks


def compute_returns(rewards, gamma):
    # rewards: (T, N) numpy array, already zeroed out past each env's episode end
    T, N = rewards.shape
    returns = np.zeros((T, N), dtype=np.float32)
    running = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def normalise_returns(returns, masks):
    """Standardise returns using only the steps that were actually played.

    Steps after an episode has ended are padding (mask 0, return 0). Including
    them in the mean and standard deviation would drag the baseline towards
    zero whenever a car finishes or leaves the track early, so they are left
    out of the statistics and come back as exactly 0.
    """
    live = masks > 0
    if not live.any():
        return np.zeros_like(returns)
    mean = returns[live].mean()
    std = returns[live].std()
    return np.where(live, (returns - mean) / (std + 1e-8), 0.0).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training on:", device, flush=True)

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    policy = PolicyNet(num_actions=envs.single_action_space.n).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    reward_history = []
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH) as f:
                reward_history = json.load(f)
        print(f"resumed from checkpoint: {len(reward_history)} prior episodes", flush=True)

    start = time.time()

    for update in range(1, NUM_UPDATES + 1):
        log_probs, entropies, rewards, masks = run_rollout(envs, policy, device, NUM_ENVS)

        rewards_arr = np.stack(rewards)          # (T, N)
        masks_arr = np.stack(masks).astype(np.float32)  # (T, N)
        episode_rewards = rewards_arr.sum(axis=0)  # (N,) total return per env this update
        reward_history.extend(episode_rewards.tolist())

        returns = normalise_returns(compute_returns(rewards_arr, GAMMA), masks_arr)
        returns_t = torch.from_numpy(returns).to(device)
        masks_t = torch.from_numpy(masks_arr).to(device)

        log_probs_t = torch.stack(log_probs)   # (T, N)
        entropy_t = torch.stack(entropies)     # (T, N)

        loss = -(masks_t * log_probs_t * returns_t).sum()
        loss = loss - ENTROPY_COEF * (masks_t * entropy_t).sum()
        loss = loss / NUM_ENVS

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_this_update = episode_rewards.mean()
        avg_recent = np.mean(reward_history[-(NUM_ENVS * 5):])
        elapsed = time.time() - start
        total_episodes = len(reward_history)
        print(
            f"update {update:4d} | episodes so far {total_episodes:5d} | "
            f"avg this batch {avg_this_update:8.1f} | avg recent {avg_recent:8.1f} | "
            f"elapsed {elapsed/60:.1f}m",
            flush=True,
        )

        if update % CHECKPOINT_EVERY == 0:
            torch.save(policy.state_dict(), CHECKPOINT_PATH)
            with open(LOG_PATH, "w") as f:
                json.dump(reward_history, f)

    torch.save(policy.state_dict(), CHECKPOINT_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(reward_history, f)
    envs.close()
    print("done")


if __name__ == "__main__":
    main()
