import torch
import torch.nn as nn

from models.base import *
from models.registry import *


class MLPBase(Policy):
    def forward(self, state_batch):
        logits = self.net(state_batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

    def get_actions(self, state_matrix, hidden=None, stochastic=True, device="cuda", threshold=0.5):
        normalized = normalize_state(state_matrix)
        state_tensor = torch.from_numpy(normalized).to(device)
        probs = self.forward(state_tensor)
        actions, log_probs = sample_actions(probs, stochastic=stochastic, threshold=threshold)
        return actions, log_probs,probs, hidden

    def init_hidden(self, batch_size=None, device=None):
        return None


class MLP1(MLPBase):
    '''
    Basic MLP with just 3 layers
    '''
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

    def _init_weights(self):
        nn.init.xavier_uniform_(self.net[0].weight)

@register_model
class MLP2(MLPBase):
    '''
    Basic MLP with just 4 layers and dropout
    '''
    def __init__(self, input_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32),nn.Tanh(),
            nn.LayerNorm(32),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.net[-1].bias, -0.2)

@register_model
class MLP3(MLPBase):
    def __init__(self, input_size=7):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.net[-1].bias, -0.2)

@register_model(history=True)
class MLP_Temp1(MLPBase):
    def __init__(self, input_size=7 , K = 5):
        super().__init__()
        self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size*K, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.net[-1].bias, -0.2)
        

# policy = MLP3() # 6 state dims + 1 action dim
# dummy_state = torch.randn(1000, 6)
# dummy_prev_action = torch.zeros(1000)
# total_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
# print(f"Total Trainable Params: {total_trainable_params}")

# # # Test the forward pass
# probs = policy.forward(dummy_state, dummy_prev_action)

# print("Dummy state mean/std: ", dummy_state.mean().item(), dummy_state.std().item())
# print("Probs shape: ", probs.shape) # Should be [1000]
# print("Probs mean/std: ", probs.mean().item(), probs.std().item())