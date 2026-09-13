import torch
import torch.nn as nn
import numpy as np

import configs


class PolicyNetwork(nn.Module):
    def __init__(self, input_size=7, hidden_size=16, gru_hidden=8):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.Dropout(p=0.1),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_size, gru_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(gru_hidden),
            nn.Dropout(p=0.1),
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Linear(gru_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # This brings the output to mean 0.5 for no bias in the start of training
        for module in self.modules():
            if isinstance(module, nn.Linear or nn.GRUCell):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.head[-1].bias,-1)
        # nn.init.zeros_(self.head[-1].weight)

    def forward(self, state_batch,prev_action, hidden):
        # state_batch: (batch, 4), hidden: (batch, gru_hidden)
        x = torch.cat([state_batch, prev_action.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        new_hidden = self.gru(x, hidden)
        logits = self.head(new_hidden).squeeze(-1)
        probs = torch.sigmoid(logits) 
        # probs = torch.clamp(probs, min=0.001, max=0.999)
        return probs, new_hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.gru_hidden, device=device)


def normalize_state(state_matrix):
    normalized = state_matrix.copy().astype(np.float32)
    normalized[:, 0] /= configs.SCREEN_HEIGHT   # bird_y
    normalized[:, 1] /= 10.0                     # velocity
    normalized[:, 2] /= configs.SCREEN_WIDTH     # pipe1_dx
    normalized[:, 3] /= configs.SCREEN_HEIGHT    # pipe1_dy
    normalized[:, 4] /= configs.SCREEN_WIDTH     # pipe2_dx
    normalized[:, 5] /= configs.SCREEN_HEIGHT    # pipe2_dy
    return normalized


def get_actions(policy, state_matrix,prev_action, hidden, stochastic=True, device="cuda", threshold=0.5):
    """
    hidden: (population, gru_hidden) tensor carried across steps within an episode.
    Returns: (actions, log_probs, new_hidden)
    """
    normalized = normalize_state(state_matrix)
    state_tensor = torch.from_numpy(normalized).to(device)

    # print(normalized[0])
    probs, new_hidden = policy(state_tensor,prev_action, hidden)

    if stochastic:
        dist = torch.distributions.Bernoulli(probs=probs)
        actions_tensor = dist.sample()
        log_probs = dist.log_prob(actions_tensor)
    else:
        actions_tensor = (probs > threshold).float()
        log_probs = None
    actions = actions_tensor.detach().cpu().numpy().astype(int)
    return actions, log_probs, new_hidden

# policy = PolicyNetwork()
# dummy_state = torch.randn(1000, 6)
# dummy_prev_action = torch.zeros(1000)
# dummy_hidden = torch.randn(1000, policy.gru_hidden)
# probs, _ = policy(dummy_state, dummy_prev_action, dummy_hidden)

# x = torch.cat([dummy_state, dummy_prev_action.unsqueeze(-1)], dim=-1)
# x = policy.input_proj(x)
# new_hidden = policy.gru(x, dummy_hidden)
# print("Dummy state mean/std: ",dummy_state.mean().item(),dummy_state.std().item())
# print("new_hidden mean/std:", new_hidden.mean().item(), new_hidden.std().item())
# print(probs.mean().item(), probs.std().item())