"""
Phase 2: fine-tune an already-good CarRacing bot to drive faster, not just
finish. Starts from checkpoint_gpu.pt and keeps training under the speed-bonus
reward from reward_wrapper.py. Saves to separate files so the original phase-1
bot is never overwritten.

Experimental and not run to completion: no phase-2 checkpoint is shipped, and
the phase-1 bot it starts from does not clear the 900-point bar (see README).
"""
import os
import json
import time
import numpy as np
import torch
import gymnasium as gym

from policy import PolicyNet
from reward_wrapper import CarRacingSpeedBonus
from train_gpu import compute_returns, normalise_returns, to_tensor_batch

# "Par" is the number of decisions a finished lap is compared against. The
# environment's own limit is 250 decisions (1000 frames), so any finished lap
# takes fewer than that. Override with phase2_config.json: {"par_decisions": N}.
PAR_DECISIONS = 230
if os.path.exists("phase2_config.json"):
    with open("phase2_config.json") as f:
        PAR_DECISIONS = json.load(f)["par_decisions"]

NUM_ENVS = 10
NUM_UPDATES = 300         # ~3000 episodes of fine-tuning
GAMMA = 0.99
LEARNING_RATE = 3e-5      # smaller than phase 1 -- refining, not relearning
MAX_DECISIONS = 250
ENTROPY_COEF = 0.005       # a bit less exploration -- it already knows how to finish
CHECKPOINT_EVERY = 10
LOG_PATH = "reward_log_phase2.json"
CHECKPOINT_PATH = "checkpoint_phase2.pt"
BASE_CHECKPOINT = "checkpoint_gpu.pt"


def make_env():
    return CarRacingSpeedBonus(gym.make("CarRacing-v3", continuous=False), PAR_DECISIONS)


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
        obs, r, terminated, truncated, _ = envs.step(action_np)
        done_now = terminated | truncated
        rewards.append(r * active)

        active = active & (~done_now)
        if not active.any():
            break

    return log_probs, entropies, rewards, masks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("phase 2 training on:", device, "| par decisions:", PAR_DECISIONS, flush=True)

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    policy = PolicyNet(num_actions=envs.single_action_space.n).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        with open(LOG_PATH) as f:
            reward_history = json.load(f)
        print(f"resumed phase 2 from its own checkpoint: {len(reward_history)} prior episodes", flush=True)
    else:
        policy.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=device))
        reward_history = []
        print("starting phase 2 from the phase-1 checkpoint", flush=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    start = time.time()

    for update in range(1, NUM_UPDATES + 1):
        log_probs, entropies, rewards, masks = run_rollout(envs, policy, device, NUM_ENVS)

        rewards_arr = np.stack(rewards)
        masks_arr = np.stack(masks).astype(np.float32)
        episode_rewards = rewards_arr.sum(axis=0)
        reward_history.extend(episode_rewards.tolist())

        returns = normalise_returns(compute_returns(rewards_arr, GAMMA), masks_arr)
        returns_t = torch.from_numpy(returns).to(device)
        masks_t = torch.from_numpy(masks_arr).to(device)

        log_probs_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)

        loss = -(masks_t * log_probs_t * returns_t).sum()
        loss = loss - ENTROPY_COEF * (masks_t * entropy_t).sum()
        loss = loss / NUM_ENVS

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recent = np.mean(reward_history[-(NUM_ENVS * 5):])
        elapsed = time.time() - start
        print(
            f"update {update:4d} | episodes so far {len(reward_history):5d} | "
            f"avg this batch {episode_rewards.mean():8.1f} | avg recent {avg_recent:8.1f} | "
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
    print("phase 2 done")


if __name__ == "__main__":
    main()
