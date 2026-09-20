import numpy as np
import torch
from src.gym import FlappyBirdEnv
import json 
import os

def compute_returns_n(rewards, gamma=0.99, order=1):
    n = len(rewards)
    returns = [0.0] * n
    G = [0.0] * (order + 1)  # G[0] unused, G[1..order] are the levels

    for t in range(n - 1, -1, -1):
        new_G = [0.0] * (order + 1)
        prev = rewards[t]
        for k in range(1, order + 1):
            new_G[k] = prev + gamma * G[k]
            prev = new_G[k]  # this level's output feeds the next level's input
        G = new_G
        returns[t] = G[order]

    return returns

def _setup(policy, lr, population_size, render, load_model=False, model_path=None):
    env = FlappyBirdEnv(population_size=population_size, render=render)
    if load_model:
        print(f"Loading pretrained policy from {model_path}....")
        try:
            policy.load(model_path)
            print(f"Loaded pretrained policy from {model_path}")
        except FileNotFoundError:
            print("No pretrained policy found, starting from scratch.")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    policy.to(device)
    return env, optimizer, scheduler, device

def _setup_episode(policy, population_size, device):
    hidden = policy.init_hidden(population_size, device)
    done = False
    log_probs_per_bird = [[] for _ in range(population_size)]
    probs_per_bird = [[] for _ in range(population_size)]
    rewards_per_bird = [[] for _ in range(population_size)]
    alive_before = np.ones(population_size, dtype=bool)
    prev_scores = np.zeros(population_size, dtype=np.int32)
    prev_action = torch.zeros(population_size, dtype=torch.float32, device=device)
    return hidden, done, log_probs_per_bird,probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action


def train(policy, num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False,load_model: bool = False, model_path: str = None, save_path: str = None):

    env, optimizer, scheduler, device = _setup(policy, lr, population_size, render, load_model, model_path)
    best_score = 10.0

    for episode in range(num_episodes):
        state = env.reset()
        hidden, done, log_probs_per_bird,probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action = _setup_episode(policy, population_size, device)

        while not done:
            actions, log_probs,probs, hidden = policy.get_actions(state, prev_action, hidden, stochastic=True, device=device)
            # hidden = hidden.detach() if torch.is_tensor(hidden) else tuple(h.detach() for h in hidden) # This should fix CUDA OOM error
            next_state, alive_after, scores, done = env.step(actions)
            prev_action = torch.from_numpy(actions).to(device).float()

            score_delta = scores - prev_scores
            prev_scores = scores.copy()

            #XXX: Reward function 
            step_reward = 0.1 + (10.0 * score_delta)
            # death_penalty = 0.5
            # newly_dead = alive_before & ~alive_after
            # step_reward = 0.1+ (12 * score_delta)- death_penalty*newly_dead.astype(np.float32)


            for i in range(population_size):
                if alive_before[i] & (log_probs is not None):  # Only record log_probs for birds that were alive this step
                    log_probs_per_bird[i].append(log_probs[i])
                    rewards_per_bird[i].append(step_reward[i])
                    probs_per_bird[i].append(probs[i])

            alive_before = alive_after
            state = next_state
        all_log_probs = []
        all_returns = []
        all_probs = []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
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
        eps = 1e-8
        entropy = -(probs_tensor * torch.log(probs_tensor + eps) + (1 - probs_tensor) * torch.log(1 - probs_tensor + eps)).mean()

        # entropy = -(log_probs_tensor * torch.exp(log_probs_tensor)).mean()
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)
            print(f"Weights saved . Best score till now : {best_score}")

        del hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action,probs_per_bird

    env.close()
    return policy

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    from models.mlp import MLP3 as policyNet

    load_model_path = None
    save_path = r"runs/mlp_with_sumofreturns.pt"

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