import numpy as np
import torch
from src.gym import FlappyBirdEnv


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    return env, optimizer, scheduler, device


def _setup_episode(policy, population_size, device):
    # init_hidden() returns a real tensor for GRUBase subclasses and None for
    # MLPBase subclasses -- policy.get_actions() handles either uniformly,
    # so this loop never needs to know which family it's driving.
    hidden = policy.init_hidden(population_size, device)
    done = False
    log_probs_per_bird = [[] for _ in range(population_size)]
    rewards_per_bird = [[] for _ in range(population_size)]
    alive_before = np.ones(population_size, dtype=bool)
    prev_scores = np.zeros(population_size, dtype=np.int32)
    prev_action = torch.zeros(population_size, dtype=torch.float32, device=device)
    return hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action


def train(policy, num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False,load_model: bool = False, model_path: str = None, save_path: str = None):

    env, optimizer, scheduler, device = _setup(policy, lr, population_size, render, load_model, model_path)
    best_score = 10.0

    for episode in range(num_episodes):
        state = env.reset()
        hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action = \
            _setup_episode(policy, population_size, device)

        while not done:
            actions, log_probs, hidden = policy.get_actions(state, prev_action, hidden, stochastic=True, device=device)
            next_state, alive_after, scores, done = env.step(actions)
            prev_action = torch.from_numpy(actions).to(device).float()

            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            #XXX: Reward function 

            step_reward = 0.1 + (10 * score_delta)

            for i in range(population_size):
                if alive_before[i] and log_probs is not None:  # Only record log_probs for birds that were alive this step
                    log_probs_per_bird[i].append(log_probs[i])
                    rewards_per_bird[i].append(step_reward[i])

            alive_before = alive_after
            state = next_state

        all_log_probs = []
        all_returns = []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
                continue
            returns_to_go = compute_returns(rewards_per_bird[i], gamma=gamma)
            all_log_probs.extend(log_probs_per_bird[i])
            all_returns.extend(returns_to_go)

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)

        # To keep exploring even when it is stuck somewhere
        entropy = -(log_probs_tensor * torch.exp(log_probs_tensor)).mean()
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)

        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

    env.close()
    return policy


if __name__ == "__main__":
    from models.gru import GRU1

    load_model_path = None
    save_path = r"runs/gru1.pt"

    #TODO : Add reward function to this 
    config = {
        "num_episodes": 800,
        "population_size": 200,
        "gamma": 0.97,
        "lr": 5e-3,
        "render": 0,
        "load_model": 0,
        "model_path": save_path,
    }
    policy = GRU1()
    trained_policy = train(policy,num_episodes=config["num_episodes"],lr=config["lr"],population_size=config["population_size"],
        gamma=config["gamma"],render=config["render"],load_model=config["load_model"],
        model_path=load_model_path,
        save_path=config["model_path"],
    )
    print(f"Saved trained policy to {save_path}")