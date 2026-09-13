import numpy as np
import torch
from gym import FlappyBirdEnv
from actor import PolicyNetwork, get_actions


def compute_returns(rewards, gamma=0.99):
    """Discounted return at each timestep, computed backward through the trajectory."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train(num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False,load_model:bool=False,model_path:str=None):
    env = FlappyBirdEnv(population_size=population_size, render=render)
    policy = PolicyNetwork()

    if load_model:
        try:
            policy.load_state_dict(torch.load(model_path))
            print(f"Loaded pretrained policy from {model_path}")
        except FileNotFoundError:
            print("No pretrained policy found, starting from scratch.")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=list(range(0, num_episodes, 100)), gamma=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # Per-bird trajectory storage: each bird's steps stop accumulating once it dies.
        log_probs_per_bird = [[] for _ in range(population_size)]
        rewards_per_bird = [[] for _ in range(population_size)]

        alive_before = np.ones(population_size, dtype=bool)
        prev_scores = np.zeros(population_size, dtype=np.int32)

        while not done:
            actions, log_probs = get_actions(policy, state, stochastic=True, device="cuda")
            next_state, alive_after, scores, done = env.step(actions)

            # Reward function
            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            # distance_to_gap = np.square(state[:, 0] - state[:, 3]*alive_after)
            # print(state[0,:])
            # distance_penalty = distance_to_gap/1e5  
            step_reward = 0.2 + (5.0 * score_delta) 

            for i in range(population_size):
                if alive_before[i] & (log_probs is not None):  # Only record log_probs for birds that were alive this step
                    log_probs_per_bird[i].append(log_probs[i])
                    rewards_per_bird[i].append(step_reward[i])

            alive_before = alive_after
            state = next_state

        # continue
        # --- Build one REINFORCE loss across all birds' trajectories ---
        all_log_probs = []
        all_returns = []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
                continue
            rewards_per_bird[i] = compute_returns(rewards_per_bird[i], gamma=gamma)
            total_reward_i = sum(rewards_per_bird[i])
            all_log_probs.extend(log_probs_per_bird[i])
            all_returns.extend([total_reward_i] * len(log_probs_per_bird[i]))

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)

        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)

        #To keep exploring even when it is stuck somewhere
        entropy = -(log_probs_tensor * torch.exp(log_probs_tensor)).mean()
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))
        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

    env.close()
    return policy


if __name__ == "__main__":
    trained_policy = train(num_episodes=1000, population_size=100,gamma=0.97, render=0,load_model=0,model_path="try2/flappy_bird_2.pt")
    save_path = r"try2/flappy_bird_3.json"
    #Save config of paramters and model settings
    config = {
        "num_episodes": 1000,
        "population_size": 100,
        "gamma": 0.97,
        "render": 0,
        "load_model": 0,
        "model_path": "try2/flappy_bird_2.pt"
    }
    config["model"] = trained_policy.state_dict()
    torch.save(config, save_path)
    torch.save(trained_policy.state_dict(), config["model_path"])
    print(f"Saved trained policy to {save_path}")
