import numpy as np
import torch

from models.base import Policy
from src.gym import FlappyBirdEnv


def _rollout(env, policy: Policy, population_size, device, threshold):
    k = policy.history
    state = env.reset()
    history = np.repeat(state[:, None, :], k, axis=1) if k else None
    hidden = policy.init_hidden(population_size, device)
    done, steps, scores = False, 0, None

    while not done:
        policy_state = history if k else state
        actions, _, _, hidden = policy.get_actions(
            policy_state, hidden, stochastic=False, device=device, threshold=threshold
        )
        state, _, scores, done = env.step(actions)
        if k:
            history = np.roll(history, -1, axis=1)
            history[:, -1, :] = state
        steps += 1
    return steps, np.asarray(scores)


def evaluate(policy: Policy, *, num_episodes=5, population_size=1, render=False,
             threshold=0.5, device=None, on_episode=None,show_score=True) -> dict:
    device = device or torch.device("cpu")
    env = FlappyBirdEnv(population_size=population_size, render=render, display_score=show_score)
    episodes = []
    policy.eval()
    try:
        with torch.no_grad():
            for i in range(num_episodes):
                steps, scores = _rollout(env, policy, population_size, device, threshold)
                ep = {"episode": i, "steps": steps,
                      "mean_score": float(np.mean(scores)), "max_score": int(np.max(scores))}
                episodes.append(ep)
                if on_episode:
                    print("\r\033[K", end="", flush=True)
                    on_episode(ep)
    finally:
        env.close()

    means = np.array([e["mean_score"] for e in episodes])
    return {
        "model": policy.registry_name,
        "num_episodes": num_episodes,
        "population_size": population_size,
        "mean_score": float(means.mean()),
        "std_score": float(means.std()),
        "max_score": max(e["max_score"] for e in episodes),
        "episodes": episodes,
    }