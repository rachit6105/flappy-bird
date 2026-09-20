import numpy as np
import torch
from src.gym import FlappyBirdEnv

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def compute_returns_n(rewards, gamma=0.99, order=1):
    """
    Generalized discounted returns.

    Each reward r_{t+m} ends up weighted by C(m+order-1, order-1) * gamma^m
    (a binomial coefficient) instead of just gamma^m — higher order stretches
    the effective credit-assignment horizon further into the future and
    inflates the return magnitude.
    """
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

def _setup(policy, lr, population_size, render,device,load_model=False, model_path=None):
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
    policy.to(device)
    return env, optimizer, scheduler

def _setup_episode(policy, population_size, device):
    hidden = policy.init_hidden(population_size, device)
    done = False
    log_probs_per_bird = [[] for _ in range(population_size)]
    probs_per_bird = [[] for _ in range(population_size)]
    rewards_per_bird = [[] for _ in range(population_size)]
    alive_before = np.ones(population_size, dtype=bool)
    prev_scores = np.zeros(population_size, dtype=np.int32)
    return hidden, done, log_probs_per_bird,probs_per_bird, rewards_per_bird, alive_before, prev_scores


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