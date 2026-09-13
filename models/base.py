import torch
import numpy as np
import game.configs as configs

def normalize_state(state_matrix):
    normalized = state_matrix.copy().astype(np.float32)
    normalized[:, 0] /= configs.SCREEN_HEIGHT   # bird_y
    normalized[:, 1] /= 10.0                     # velocity
    normalized[:, 2] /= configs.SCREEN_WIDTH     # pipe1_dx
    normalized[:, 3] /= configs.SCREEN_HEIGHT    # pipe1_dy
    normalized[:, 4] /= configs.SCREEN_WIDTH     # pipe2_dx
    normalized[:, 5] /= configs.SCREEN_HEIGHT    # pipe2_dy
    return normalized


def sample_actions(probs: torch.Tensor, stochastic=True, threshold=0.5):
    if stochastic:
        dist = torch.distributions.Bernoulli(probs=probs)
        actions_tensor = dist.sample()
        log_probs = dist.log_prob(actions_tensor)
    else:
        actions_tensor = (probs > threshold).float()
        log_probs = None

    actions = actions_tensor.detach().cpu().numpy().astype(int)
    return actions, log_probs