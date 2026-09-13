import numpy as np
import torch

from gym import FlappyBirdEnv
from actor import PolicyNetwork, get_actions


def test(model_path, population_size=1, render=True, num_episodes=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FlappyBirdEnv(population_size=population_size, render=render)
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    policy.eval()  # no-op for this architecture (no dropout/batchnorm), but good habit

    episode_scores = np.array([], dtype=np.float32)

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            steps = 0

            # Reset recurrent state and prev_action fresh at the start of every episode —
            # a bird's memory shouldn't carry over from the last episode's death.
            hidden = policy.init_hidden(population_size, device)
            prev_action = torch.zeros(population_size, dtype=torch.float32, device=device)

            while not done:
                # stochastic=False -> deterministic threshold, no random sampling.
                # This is the "actually play well" mode, as opposed to training-time exploration.
                actions, _, hidden = get_actions(
                    policy, state, prev_action, hidden,
                    stochastic=False, device=device, threshold=0.65,
                )
                state, alive, scores, done = env.step(actions)
                steps += 1

            mean_score = float(np.mean(scores))
            episode_scores = np.append(episode_scores, mean_score)
            mean_score = episode_scores.mean()
            max_score = int(np.max(scores))
            print(f"Episode {episode:3d} | steps={steps:5d} | mean_score={mean_score:6.2f} | max_score={max_score:3d}")

    env.close()

    print(f"\nOverall mean score across {num_episodes} episodes: {np.mean(episode_scores):.2f}")
    return episode_scores


if __name__ == "__main__":
    test(
        model_path="try_gru/gru_2pipe_1.pt",
        population_size=1,   # single bird for clean visual evaluation
        render=False,
        num_episodes=100,
    )

# Current best : Overall mean score across 100 episodes: 61.52
# Current best max score : 298
#Using GRU and 6 states