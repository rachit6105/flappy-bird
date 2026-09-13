import torch
import torch.nn as nn
import numpy as np

import configs


class PolicyNetwork(nn.Module):

    def __init__(self, input_size=4, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight)

    def forward(self, state_batch):
        # state_batch: (batch, 4) tensor -> returns (batch,) flap probabilities in [0, 1]
        logits = self.net(state_batch).squeeze(-1)
        return torch.sigmoid(logits)
        # return logits


def normalize_state(state_matrix):
    """
    state_matrix: (population, 4) numpy array -> [y, velocity, pipe_dx, pipe_dy]
    Raw values live on very different scales (pixels vs velocity units), which
    slows/destabilizes training. Scale everything to roughly [-1, 1] / [0, 1].
    """
    normalized = state_matrix.copy().astype(np.float32)
    normalized[:, 0] /= configs.SCREEN_HEIGHT       # bird_y
    normalized[:, 1] /= 10.0                         # velocity, rough scale for this game's gravity/flap magnitude
    normalized[:, 2] /= configs.SCREEN_WIDTH         # pipe_dx
    normalized[:, 3] /= configs.SCREEN_HEIGHT        # pipe_dy
    return normalized


def get_actions(policy, state_matrix, stochastic=True, device="cuda",threshold=0.5):
    """
    state_matrix: (population, 4) numpy array from env.step()/reset()
    stochastic=True: sample from Bernoulli(prob) -> use during training (exploration)
    stochastic=False: argmax/threshold at 0.5 -> use during evaluation/inference

    Returns: (actions, log_probs)
        actions: (population,) numpy int array of 0/1, feed directly into env.step()
        log_probs: (population,) torch tensor, needed later for REINFORCE-style loss.
                   Ignore this return value entirely if you're just doing inference.
    """
    normalized = normalize_state(state_matrix)
    state_tensor = torch.from_numpy(normalized).to(device)

    logits = policy(state_tensor)  # (population,)

    if stochastic:
        dist = torch.distributions.Bernoulli(probs=logits)
        actions_tensor = dist.sample()
        log_probs = dist.log_prob(actions_tensor)
    else:
        actions_tensor = (torch.sigmoid(logits) > threshold).float()
        log_probs = None

    actions = actions_tensor.detach().cpu().numpy().astype(int)
    return actions, log_probs

