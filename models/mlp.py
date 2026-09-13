import torch
import torch.nn as nn

from base import normalize_state, sample_actions


class MLPBase(nn.Module):
    def forward(self, state_batch, prev_action=None):
        logits = self.net(state_batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, None

    def get_actions(self, state_matrix, prev_action, hidden, stochastic=True, device="cuda", threshold=0.5):
        normalized = normalize_state(state_matrix)
        state_tensor = torch.from_numpy(normalized).to(device)

        probs, new_hidden = self.forward(state_tensor, prev_action, hidden)
        actions, log_probs = sample_actions(probs, stochastic=stochastic, threshold=threshold)
        return actions, log_probs, new_hidden

    def init_hidden(self, batch_size, device):
        # Stateless model -- returns None so train.py can call this
        # uniformly across every family without an isinstance/hasattr check.
        return None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


class MLP1(MLPBase):
    '''
    Basic MLP with just 3 layers
    '''
    def __init__(self, input_size=7, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.net[0].weight)