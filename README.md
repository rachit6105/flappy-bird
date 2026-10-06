# Flappy Bird

This repository is an experimentation project for training and comparing reinforcement-learning policies on Flappy Bird. It includes a custom Pygame environment, registered model families such as MLP, GRU, and LSTM, and saved policy checkpoints. The `flappy-bird` command-line tool lets you watch a model play or evaluate its scores over multiple episodes; you can also launch the keyboard-controlled game and play yourself. The gameplay video below shows the MLP policy in action.We achieved best result of **6143** using temporal mlp model.

<div align="center">
  
  https://github.com/user-attachments/assets/2baae41b-4dfc-43ca-bd2d-c42f2d748da4
  
</div>


## Setup

Clone the repository and change into its directory:

```bash
git clone https://github.com/rachit6105/flappy-bird.git
cd flappy-bird
```

Install [uv](https://docs.astral.sh/uv/) if it is not already available, then install the project dependencies:

```bash
uv sync
```

## Play the game

Launch the keyboard-controlled Pygame game from the repository root:

```bash
cd game && ../.venv/bin/python main.py
```

Press **Space** to start and flap. After a collision, press **Escape** to restart. Close the game window to exit.

## Watch an agent play

The `flappy-bird` command-line tool provides commands for playing and evaluating policies. The `play` command renders the selected policy by default and loads its checkpoint from `runs/best_models.json`:

```bash
flappy-bird play gru
```

### Best models

The `runs/best_models.json` manifest records the selected model and checkpoint for each family, along with any saved settings:

| Family | Model | Checkpoint | Recorded settings |
|---|---|---|---|
| GRU | `gru.GRU1` | `runs/gru1.pt` | `K=1`, `thr=0.65` |
| MLP | `mlp.MLP_Temp1` | `runs/mlp_temporal.pt` | `K=50` |

List registered model families and checkpoints with:

```bash
flappy-bird list
```

To see all available commands and options:

```bash
flappy-bird --help
```

## Evaluate a saved policy

Evaluation runs headlessly by default. For example, evaluate the registered GRU policy over 50 episodes:

```bash
flappy-bird eval gru --episodes 50
```

You can also provide a checkpoint explicitly:

```bash
flappy-bird eval gru --checkpoint runs/gru1.pt
```

Add `--render` to show the game during evaluation. Evaluation reports the mean score, score deviation, best score, and episode count.

### Recorded result

A 50-episode evaluation recorded a mean score of **1761.52 ± 1372.54**, with a best score of **6143**. Results vary between runs.

## Project structure

```text
flappy_bird/
├── main.py             # Policy play/evaluation CLI
├── game/               # Keyboard-controlled Pygame game
├── models/             # Policy models and registry
├── src/                # Environment and evaluation code
├── runs/               # Checkpoints, model configs, and best_models.json
├── pyproject.toml
└── README.md
```
