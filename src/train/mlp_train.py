import json
import os

import numpy as np
import torch

from src.train import utils


def train(policy, num_episodes=500, population_size=100, gamma=0.98, lr=5e-3,
          render=False, load_model=False, model_path=None, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, optimizer, scheduler = utils._setup(
        policy, lr, population_size, render, device, load_model, model_path
    )
    best_score = 10.0

    for episode in range(num_episodes):
        state = np.column_stack((env.reset(), np.zeros(population_size)))
        hidden, done, log_probs_per_bird, probs_per_bird, rewards_per_bird, alive_before, prev_scores = (
            utils._setup_episode(policy, population_size, device)
        )

        while not done:
            actions, log_probs, probs, hidden = policy.get_actions(
                state, hidden, stochastic=True, device=device
            )
            state, alive_after, scores, done = env.step(actions)
            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            step_reward = 0.1 + (10.0 * score_delta)

            for bird_index in range(population_size):
                if alive_before[bird_index] and log_probs is not None:
                    log_probs_per_bird[bird_index].append(log_probs[bird_index])
                    rewards_per_bird[bird_index].append(step_reward[bird_index])
                    probs_per_bird[bird_index].append(probs[bird_index])
            alive_before = alive_after

        all_log_probs = []
        all_returns = []
        all_probs = []
        for bird_index in range(population_size):
            if not rewards_per_bird[bird_index]:
                continue
            # rewards_per_bird[i] = compute_returns(rewards_per_bird[i], gamma=gamma)
            # returns_to_go = compute_returns_n(rewards_per_bird[i], gamma=gamma,order=3)
            # all_returns.extend(returns_to_go)
            total_reward_i = sum(rewards_per_bird[i])
            all_log_probs.extend(log_probs_per_bird[i])
            all_returns.extend([total_reward_i] * len(log_probs_per_bird[i]))
            all_probs.extend(probs_per_bird[i])

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)
        probs_tensor = torch.stack(all_probs)
        entropy = -(probs_tensor * torch.log(probs_tensor + 1e-8) +
                    (1 - probs_tensor) * torch.log(1 - probs_tensor + 1e-8)).mean()
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))
        print(
            f"Episode {episode:4d} | mean_score={mean_score:6.2f} | "
            f"max_score={max_score:3d} | loss={loss.item():8.4f}"
        )

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)
            print(f"Weights saved . Best score till now : {best_score}")

    env.close()
    return policy


if __name__ == "__main__":
    from models.mlp import MLP3

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = {
        "num_episodes": 1000,
        "population_size": 200,
        "gamma": 0.97,
        "lr": 5e-3,
        "render": 0,
        "load_model": 0,
        "model_path": "runs/mlp_with_sumofreturns.pt",
    }
    config_path = os.path.splitext(config["model_path"])[0] + "_config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
    train(MLP3(), save_path=config["model_path"], **config)
