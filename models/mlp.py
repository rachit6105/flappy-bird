import torch
import torch.nn as nn

from models.base import normalize_state, sample_actions


class MLPBase(nn.Module):
    def forward(self, state_batch,prev_action=None):
        x = torch.cat([state_batch, prev_action.unsqueeze(-1)], dim=-1)
        logits = self.net(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

    def get_actions(self, state_matrix, prev_action, hidden=None, stochastic=True, device="cuda", threshold=0.5):
        normalized = normalize_state(state_matrix)
        state_tensor = torch.from_numpy(normalized).to(device)
        probs = self.forward(state_tensor, prev_action=prev_action)
        actions, log_probs = sample_actions(probs, stochastic=stochastic, threshold=threshold)
        return actions, log_probs,probs, hidden

    def init_hidden(self, batch_size=None, device=None):
        return None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))


class MLP1(MLPBase):
    '''
    Basic MLP with just 3 layers
    '''
    def __init__(self, input_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.net[0].weight)

class MLP2(MLPBase):
    '''
    Basic MLP with just 4 layers and dropout
    '''
    def __init__(self, input_size=7, layers=[16, 32, 16, 8, 4]):
        super().__init__()
        modules = []
        in_features = input_size
        for out_features in layers:
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.Tanh())
            # Optional: Add nn.LayerNorm(out_features) or nn.Dropout(0.1) here if needed
            in_features = out_features
            
        # Final output layer mapping to 1
        modules.append(nn.Linear(in_features, 1))
        
        # Unpack the list into nn.Sequential
        self.net = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear or nn.GRUCell)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)