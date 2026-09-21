import torch
import torch.nn as nn
from models.base import normalize_state, sample_actions


class LSTMBase(nn.Module):
    """
    Base class for LSTM-based (recurrent) policies. Subclasses (LSTM1,
    LSTMWithLayerNorm, ...) define their own __init__ -- building
    self.input_proj, self.lstm, self.head, setting self.lstm_hidden, and
    calling _init_weights -- while forward, get_actions, init_hidden,
    save/load stay shared here and shouldn't need touching per-variant.

    Contract for subclasses:
        self.lstm_hidden : int, size of the recurrent hidden state
        self.input_proj  : nn.Module, (batch, input_size) -> (batch, proj_size)
        self.lstm        : nn.LSTMCell, (batch, proj_size), (h, c)
                               -> (h_new, c_new), each (batch, lstm_hidden)
        self.head        : nn.Module, (batch, lstm_hidden) -> (batch, 1) raw logits
    """

    def init_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def get_actions(self, state_matrix, hidden, stochastic=True, device="cuda", threshold=0.5):
        normalized = normalize_state(state_matrix)
        state_tensor = torch.from_numpy(normalized).to(device)

        probs, new_hidden = self.forward(state_tensor, hidden)
        actions, log_probs = sample_actions(probs, stochastic=stochastic, threshold=threshold)
        return actions, log_probs, probs, new_hidden

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


class LSTM1(LSTMBase):

    def __init__(self, input_size=7, hidden_size=16, lstm_hidden=8):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Tanh(),
        )
        self.lstm = nn.LSTMCell(hidden_size, lstm_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1),
        )
        self._init_weights()

    def forward(self, state_batch, hidden):
        x = self.input_proj(state_batch)
        h, c = hidden
        h_new, c_new = self.lstm(x, (h, c))
        logits = self.head(h_new).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, (h_new, c_new)

    def _init_weights(self):
        '''
        Same rationale as GRU1: zero-init biases so initial flap probability
        starts near 0.5 rather than skewed, avoiding a long "flap constantly"
        phase before the policy learns to be selective.
        '''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.LSTMCell):
                nn.init.xavier_uniform_(module.weight_hh)
                # LSTMCell packs 4 gates (i, f, g, o) into one bias vector of
                # size 4*hidden. Set the forget-gate bias slice to 1 (standard
                # LSTM init trick -- start with "remember by default" rather
                # than a neutral/random forget gate) instead of zeroing it.
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)
                # start, end = module.hidden_size, 2 * module.hidden_size
                # nn.init.constant_(module.bias_ih[start:end], 1.0)
                # nn.init.constant_(module.bias_hh[start:end], 1.0)
        nn.init.constant_(self.head[-1].bias, -1)


# policy = LSTM1()  # 7 state dims + 1 action dim
# dummy_state = torch.randn(1000, 6)
# dummy_prev_action = torch.zeros(1000)
# dummy_hidden = policy.init_hidden(1000, device="cpu")  # (h, c) tuple

# total_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
# print(f"Total Trainable Params: {total_trainable_params}")

# # Test the forward pass
# probs, new_hidden = policy.forward(dummy_state, dummy_prev_action, dummy_hidden)

# print("Dummy state mean/std: ", dummy_state.mean().item(), dummy_state.std().item())
# # print("Probs shape: ", probs.shape)  # Should be [1000]
# print("Probs mean/std: ", probs.mean().item(), probs.std().item())
# print("Probs min/max: ", probs.min().item(), probs.max().item())
