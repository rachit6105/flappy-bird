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

def _setup(lr,population_size,render,load_model:bool=False,model_path:str=None):
    env = FlappyBirdEnv(population_size=population_size, render=render)
    policy = PolicyNetwork()
    if load_model:
        print(f"Loading pretrained policy from {model_path}....")
        try:
            policy.load_state_dict(torch.load(model_path))
            print(f"Loaded pretrained policy from {model_path}")
        except FileNotFoundError:
            print("No pretrained policy found, starting from scratch.")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    return env, policy, optimizer,scheduler, device

def _setup_episode(policy, population_size, device):
    hidden = policy.init_hidden(population_size, device)
    done = False
    log_probs_per_bird = [[] for _ in range(population_size)]
    rewards_per_bird = [[] for _ in range(population_size)]
    alive_before = np.ones(population_size, dtype=bool)
    prev_scores = np.zeros(population_size, dtype=np.int32)
    prev_action = torch.zeros(population_size, dtype=torch.float32, device=device)
    return hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action


def train(num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False,
          load_model:bool=False,model_path:str=None,save_path:str=None):

    env, policy, optimizer,scheduler, device = _setup(lr,population_size,render,load_model,model_path)
    best_score = 10.0

    for episode in range(num_episodes):
        state = env.reset()
        hidden, done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action = _setup_episode(policy, population_size, device)

        while not done:
            actions, log_probs, hidden = get_actions(policy, state,prev_action ,hidden, stochastic=True, device="cuda")
            next_state, alive_after, scores, done = env.step(actions)
            prev_action = torch.from_numpy(actions).to(device).float()  
            # Reward function
            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            # distance_to_gap = np.square(state[:, 0] - state[:, 3]*alive_after)
            step_reward = 0.1 + (10* score_delta) 
            # - ((alive_before^alive_after).sum())/population_size 

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
            returns_to_go = compute_returns(rewards_per_bird[i], gamma=gamma)
            all_log_probs.extend(log_probs_per_bird[i])
            all_returns.extend(returns_to_go)
            # total_reward_i = sum(rewards_per_bird[i])
            # all_log_probs.extend(log_probs_per_bird[i])
            # all_returns.extend([total_reward_i] * len(log_probs_per_bird[i]))

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)

        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)

        #To keep exploring even when it is stuck somewhere
        entropy = -(log_probs_tensor * torch.exp(log_probs_tensor)).mean()
        # print(entropy.item())
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.01 

        optimizer.zero_grad()
        loss.backward()
        # grad_norms = [p.grad.norm().item() for p in policy.parameters() if p.grad is not None]
        # print("Grad mean:", np.mean(grad_norms)/population_size)/
        # print("Grad max:", np.max(grad_norms))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0) 
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        if mean_score > best_score :
            best_score = mean_score
            torch.save(policy.state_dict(), save_path)
        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

    env.close()
    return policy


if __name__ == "__main__":
    # load_model_path = r"try_gru/gru_2pipe_1.pt"
    load_model_path = None
    save_path = r"try_gru/gru_2pipe_2.json"
    #Save config of paramters and model settings
    config = {
        "num_episodes": 800,
        "population_size": 200,
        "gamma": 0.97,
        "lr" : 5e-3,
        "render": 0,
        "load_model": 0,
        "model_path": "try_gru/gru_2pipe_2.pt"
    }
    trained_policy = train(num_episodes=config["num_episodes"],lr=config["lr"], population_size=config["population_size"],
                           gamma=config["gamma"],render=config["render"],
                           load_model=config["load_model"],model_path=load_model_path,save_path = config["model_path"])
    config["model"] = save_path
    torch.save(config, save_path)
    print(f"Saved trained policy to {save_path}")
