import json
import numpy as np
import torch
import json 
import os
from src.train import utils


def train(policy, num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False,load_model: bool = False, model_path: str = None, save_path: str = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, optimizer, scheduler = utils._setup(policy, lr, population_size, render, device, load_model, model_path)
    best_score = 10.0

    for episode in range(num_episodes):
        state = env.reset()
        state = np.column_stack((state,np.zeros(population_size)))
        hidden, done, log_probs_per_bird,probs_per_bird, rewards_per_bird, alive_before, prev_scores = utils._setup_episode(policy, population_size, device)

        while not done:
            actions, log_probs,probs, hidden = policy.get_actions(state, hidden, stochastic=True, device=device)
            # hidden = hidden.detach() if torch.is_tensor(hidden) else tuple(h.detach() for h in hidden) # This should fix CUDA OOM error
            next_state, alive_after, scores, done = env.step(actions)
            score_delta = scores - prev_scores
            prev_scores = scores.copy()

            step_reward = 0.1 + (10.0 * score_delta)

            for i in range(population_size):
                if alive_before[i] & (log_probs is not None):  # Only record log_probs for birds that were alive this step
                    log_probs_per_bird[i].append(log_probs[i])
                    rewards_per_bird[i].append(step_reward[i])

            alive_before = alive_after
            state = next_state
        all_log_probs = []
        all_returns = []
        # all_probs = []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
                continue
            rewards_per_bird[i] = utils.compute_returns(rewards_per_bird[i], gamma=gamma)
            returns_to_go = utils.compute_returns(rewards_per_bird[i], gamma=gamma)
            all_log_probs.extend(log_probs_per_bird[i])
            all_returns.extend(returns_to_go)
            # all_probs.extend(probs_per_bird[i])

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)

        # probs_tensor = torch.stack(all_probs)
        # eps = 1e-8
        # entropy = -(probs_tensor * torch.log(probs_tensor + eps) + (1 - probs_tensor) * torch.log(1 - probs_tensor + eps)).mean()

        entropy = -(log_probs_tensor * torch.exp(log_probs_tensor)).mean()
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)
            print(f"Weights saved . Best score till now : {best_score}")

        del hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores,probs_per_bird

    env.close()
    return policy

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from models.gru import GRU1 as policyNet

if __name__ == "__main__":
        
    load_model_path = None
    save_path = r"runs/gru_new.pt"

    #TODO : Add reward function to this 
    config = {
        "num_episodes": 1000,
        "population_size": 200,
        "gamma": 0.97,
        "lr": 5e-3,
        "render": 0,
        "load_model": 0,
        "model_path": save_path,
    }
    policy = policyNet()
    config_path = os.path.splitext(config["model_path"])[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    trained_policy = train(policy,num_episodes=config["num_episodes"],lr=config["lr"],population_size=config["population_size"],
        gamma=config["gamma"],render=config["render"],load_model=config["load_model"],
        model_path=load_model_path,
        save_path=config["model_path"],
    )
    print(f"Saved trained policy to {save_path}")