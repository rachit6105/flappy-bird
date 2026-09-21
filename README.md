# Flappy Bird RL Benchmark

A lightweight reinforcement learning project for benchmarking different models on the classic Flappy Bird environment.

## Overview

This repository is built to compare how different neural network policies perform in the same Flappy Bird task. The current setup includes:

- a feedforward MLP baseline
- a recurrent GRU policy
- a custom game environment
- training and evaluation scripts
- saved model checkpoints in the `runs/` directory

The goal is to test which model learns the game best under the same environment and reward setup.

## Project Structure

```text
flappy_bird/
├── README.md
├── main.py
├── training.py
├── eval.py
├── pyproject.toml
├── game/
├── models/
├── src/
├── runs/
└── .venv/
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/rachit6105/flappy-bird
cd flappy_bird
```

### 2. Create and sync the environment with uv

```bash
uv sync
```

This creates the project environment and installs the dependencies from `pyproject.toml`.

### 3. Activate the environment

```bash
source .venv/bin/activate
```

If you prefer to run commands directly through uv without activating the environment, use:

```bash
uv run training.py
uv run eval.py
```

### 4. Run training and evaluation

```bash
uv run training.py
uv run eval.py
```


## Game Environment

The environment is implemented in `src/gym.py` and uses a custom Pygame-based Flappy Bird setup. The agent controls a bird that must pass through gaps between pipes while avoiding collisions. The environment exposes a structured state containing:

- bird position
- bird velocity
- upcoming pipe information

The action space is binary:

- 0 = do nothing
- 1 = flap

## Training Approach

Training is performed with a policy-gradient style update in `training.py`.

The model receives a state, samples an action, and gets a reward based on progress through the game. The return is computed using discounted rewards, normalized, and used to update the policy. The project also includes entropy regularization to encourage exploration and save the best-performing checkpoint.

## Models

### MLP

The MLP model is a simple feedforward baseline. It processes the current state and decides whether to flap or not. It is fast and useful as a comparison baseline.

### GRU

The GRU model is designed for sequential decision making. It keeps a hidden state across time steps, which helps it learn temporal patterns in gameplay and react more effectively to pipe spacing and motion.

## Current Results

The repository currently includes trained GRU checkpoints in `runs/`:

- `gru1.pt`
- `gru2.pt`
- `gru3.pt`
- `gru4.pt`
- `gru_max2k.pt`

The evaluation script reports strong GRU performance, with an example mean score around 61.52 across 100 episodes and a best max score of 298 in the current benchmark comments.

## Notes

This project is intended as a model comparison and experimentation workspace. It is set up so additional architectures can be added and evaluated under the same conditions.

## License

This project is for educational and research use.
