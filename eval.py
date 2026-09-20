import numpy as np
import torch

from src.gym import FlappyBirdEnv
from models.gru import GRU1


def test(model_path, threshold,population_size=1, render=True, num_episodes=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FlappyBirdEnv(population_size=population_size, render=render)
    policy = GRU1()
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    policy.eval()  # no-op for this architecture (no dropout/batchnorm), but good habit

    episode_scores = np.array([], dtype=np.float32)

    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            steps = 0

            hidden = policy.init_hidden(population_size, device)
            prev_action = torch.zeros(population_size, dtype=torch.float32, device=device)

            while not done:
                actions, _,_, hidden = policy.get_actions(
                    state, prev_action, hidden,
                    stochastic=False, device=device, threshold=threshold,
                )
                state, _, scores, done = env.step(actions)
                steps += 1

            mean_score = float(np.mean(scores))
            episode_scores = np.append(episode_scores, mean_score)
            mean_score = episode_scores.mean()
            max_score = int(np.max(scores))
            # print("\r\033[K", end="", flush=True)       
            print(f"Episode {episode:3d} | steps={steps:5d} | mean_score={mean_score:6.2f} | max_score={max_score:3d}")

    env.close()
    print(f"\nOverall mean score across {num_episodes} episodes: {np.mean(episode_scores):.2f}")
    return episode_scores


if __name__ == "__main__":
    # for i in range(60,70,1):
        # print(f"Testing for  = {i/100}")
    test(model_path="runs/gru_og_reptest.pt",population_size=1,threshold=0.65 ,render=0,num_episodes=10)

# Current best : Overall mean score across 30 episodes: 739.69
# Current best max score : 2101
#Using GRU and 6 states