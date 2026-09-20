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

def pwl_slope(x,schedule):
    """
    schedule: A list of (x, m) tuples where m is the slope starting at step x.
    Returns a callable function to get the accumulated value at any step.
    """
    # schedule = sorted(schedule, key=lambda p: p[0])
    y = 0.0
    for i in range(len(schedule)):
        x_start, m = schedule[i]
        # The interval ends at the next schedule point, or infinity if it's the last point
        x_end = schedule[i+1][0] if i + 1 < len(schedule) else float('inf')
        
        if x > x_start:
            # Calculate how many steps overlap with this specific slope's interval
            steps_in_range = min(x, x_end) - x_start
            y += m * steps_in_range
        else:
            break # We haven't reached this x yet
    return float(y)

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


def train(policy, num_episodes=500, population_size=100, gamma=0.98, lr=5e-3, render=False, load_model=False, model_path=None, save_path=None, K=4):

    env, optimizer, scheduler, device = _setup(policy, lr, population_size, render, load_model, model_path)
    best_score = 10.0
    schedule_slopes = [
        (0, 0.0),      # Slope is 0 from step 0
        (300, 0.001),  # Slope becomes 0.001 starting at step 200
        (800, 0.0)     # Slope becomes 0 again starting at step 210
    ]
    for episode in range(num_episodes):
        state = env.reset()
        state = np.column_stack((state,np.zeros(population_size)))
        # Initialize Frame Stack buffer by repeating the first frame K times
        # Shape: (population_size, K, 7)
        state_history = np.repeat(state[:, np.newaxis, :], K, axis=1)
        steps = np.zeros(population_size,dtype=np.int32)
        _,done, log_probs_per_bird, probs_per_bird, rewards_per_bird, alive_before, prev_scores,_ = _setup_episode(policy,population_size,device)

        while not done:
            # Flatten the (K, 7) buffer into a single (K*7) vector for the MLP            
            # Removed hidden and prev_action
            actions, log_probs, probs,_ = policy.get_actions(state_history, stochastic=True, device=device)
            next_state, alive_after, scores, done = env.step(actions)
            next_state = np.column_stack((next_state, actions))
            # print(next_state[0])
            # Shift the history buffer left by 1 and insert the newest state at the end
            state_history = np.roll(state_history, shift=-1, axis=1)
            state_history[:, -1, :] = next_state

            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            steps+=1
            # step_reward = 0.1 + (10.0 * score_delta)
            death_penalty = np.array([pwl_slope(step,schedule_slopes) for step in steps])
            newly_dead = alive_before & ~alive_after
            step_reward = 0.1+ (12 * score_delta)- death_penalty*newly_dead.astype(np.float32)


            for i in range(population_size):
                if alive_before[i] & (log_probs is not None):  # Only record log_probs for birds that were alive this step
                    log_probs_per_bird[i].append(log_probs[i])
                    rewards_per_bird[i].append(step_reward[i])
                    probs_per_bird[i].append(probs[i])

            alive_before = alive_after
            
        all_log_probs = []
        all_returns = []
        all_probs = []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
                continue
            
            # Use proper discounted returns
            # returns_to_go = compute_returns_n(rewards_per_bird[i], gamma=gamma,order=2)
            # rewards_per_bird[i] = compute_returns(rewards_per_bird[i], gamma=gamma)
            returns_to_go = compute_returns_n(rewards_per_bird[i], gamma=gamma,order=1)
            all_returns.extend(returns_to_go)
            # total_reward_i = sum(rewards_per_bird[i])
            all_log_probs.extend(log_probs_per_bird[i])
            # all_returns.extend([total_reward_i] * len(log_probs_per_bird[i]))
            all_probs.extend(probs_per_bird[i])
            

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(all_log_probs)

        probs_tensor = torch.stack(all_probs)
        eps = 1e-8
        entropy = -(probs_tensor * torch.log(probs_tensor + eps) + (1 - probs_tensor) * torch.log(1 - probs_tensor + eps)).mean()

        # Increased entropy slightly to prevent premature MLP collapse
        loss = -(log_probs_tensor * returns_tensor).mean() - entropy * 0.02

        optimizer.zero_grad()
        loss.backward()
        
        # total_norm = 0.0
        # print("\n--- Gradient Sizes ---")
        # for name, param in policy.named_parameters():
        #     if param.grad is not None:
        #         param_norm = param.grad.data.norm(2).item()
        #         total_norm += param_norm ** 2
        #         grad_max = param.grad.data.max().item()
        #         grad_min = param.grad.data.min().item()
        #         print(f"{name:30s} | Norm: {param_norm:8.4f} | Min: {grad_min:8.4f} | Max: {grad_max:8.4f}")
        # total_norm = total_norm ** 0.5
        # print(f"{'Total Combined Norm':30s} | Norm: {total_norm:8.4f}")
        # print("----------------------\n")
        
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        
        optimizer.step()
        scheduler.step()

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | loss={loss.item():8.4f}")

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)
            print(f"Weights saved . Best score till now : {best_score}")

        del done, log_probs_per_bird, rewards_per_bird, alive_before, prev_scores, probs_per_bird

    env.close()
    return policy

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    from models.mlp import MLP_Temp1 as policyNet

    load_model_path = r"runs/mlp_frame_stack.pt"
    save_path = r"runs/mlp_frame_stack_cont.pt"

    config = {"num_episodes": 1000,"population_size": 100,"gamma": 0.99,"lr": 3e-3,"render": 0,"load_model": 1,"model_path": save_path,"K": 50 }
    
    policy = policyNet(input_size=7 , K = config["K"]) 
    config_path = os.path.splitext(config["model_path"])[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    trained_policy = train(policy,num_episodes=config["num_episodes"],lr=config["lr"],population_size=config["population_size"],gamma=config["gamma"],
                           render=config["render"],load_model=config["load_model"],model_path=load_model_path,save_path=config["model_path"],K=config["K"])
    print(f"Saved trained policy to {save_path}")