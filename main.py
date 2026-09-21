import argparse

import numpy as np
import torch

from models.gru import GRU1
from models.lstm import LSTM1
from models.mlp import MLP_Temp1
from src.gym import FlappyBirdEnv

#TODO : Change this to support any model type from any model family
def _build_policy(model_name, model_path, device, k):
    if model_name == "gru":
        policy = GRU1()
    elif model_name == "lstm":
        policy = LSTM1()
    else:
        policy = MLP_Temp1(K=k)

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.to(device)
    policy.eval()
    return policy


def evaluate(model_name, model_path, render=False, k=1, population_size=1,num_episodes=5, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FlappyBirdEnv(population_size=population_size, render=render,display_score=True)
    policy = _build_policy(model_name, model_path, device, k)
    episode_scores = []

    try:
        with torch.no_grad():
            for episode in range(num_episodes):
                state = env.reset()
                state_history = np.repeat(state[:, None, :], k, axis=1)
                hidden = policy.init_hidden(population_size, device)
                done = False
                steps = 0

                while not done:
                    policy_state = state_history if model_name == "mlp" else state
                    actions, _, _, hidden = policy.get_actions(policy_state,hidden,stochastic=False,device=device,threshold=threshold)
                    state, _, scores, done = env.step(actions)
                    if model_name == "mlp":
                        state_history = np.roll(state_history, -1, axis=1)
                        state_history[:, -1, :] = state
                    steps += 1

                mean_score = float(np.mean(scores))
                episode_scores.append(mean_score)
                print("\r\033[K", end="", flush=True)   
                print(
                    f"Episode {episode:3d} | steps={steps:5d} | "
                    f"mean_score={np.mean(episode_scores):6.2f} | "
                    f"max_score={int(np.max(scores)):3d}"
                )
    finally:
        env.close()
        print("Game ended")

    return np.asarray(episode_scores, dtype=np.float32)


def _parser(model=None):
    parser = argparse.ArgumentParser(description="Evaluate a Flappy Bird policy.")
    if model is None:
        parser.add_argument("--model", choices=("gru", "lstm", "mlp"), required=True)
    parser.add_argument("--model-name", required=True, help="Path to the model checkpoint.")
    parser.add_argument("-r", action="store_true", help="Render the game window.")
    parser.add_argument("-K", type=int, default=1, help="Frame history length for MLP models.")
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--thr", type=float, default=0.5)
    return parser


def _run(model=None):
    args = _parser(model).parse_args()
    model_name = model or args.model
    return evaluate(
        model_name=model_name,
        model_path=args.model_name,
        render=args.r,
        k=args.K if model_name == "mlp" else 1,
        population_size=args.population_size,
        num_episodes=args.num_episodes,
        threshold=args.thr,
    )


def main():
    return _run()


def main_mlp():
    return _run("mlp")


def main_gru():
    return _run("gru")


def main_lstm():
    return _run("lstm")


if __name__ == "__main__":
    main()
