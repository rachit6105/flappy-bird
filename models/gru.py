import torch
import torch.nn as nn
from models.base import *
from models.registry import *


class GRUBase(Policy):
    """
    Contract for subclasses:
        self.gru_hidden : int, size of the recurrent hidden state
        self.input_proj : nn.Module, (batch, input_size) -> (batch, proj_size)
        self.gru        : nn.GRUCell, (batch, proj_size), (batch, gru_hidden)
                              -> (batch, gru_hidden)
        self.head       : nn.Module, (batch, gru_hidden) -> (batch, 1) raw logits
    """


    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.gru_hidden, device=device)

    def get_actions(self, state_matrix, hidden, stochastic=True, device="cuda", threshold=0.5):
        """
        hidden: (population, gru_hidden) tensor carried across steps within
        an episode -- caller is responsible for threading new_hidden back in
        on the next call (unlike stateless families, this one matters).

        Returns: (actions, log_probs, new_hidden)
        """
        normalized = normalize_state(state_matrix)
        state_tensor = torch.from_numpy(normalized).to(device)
        probs, new_hidden = self.forward(state_tensor, hidden)
        actions, log_probs = sample_actions(probs, stochastic=stochastic, threshold=threshold)
        return actions, log_probs,probs, new_hidden


@register_model
class GRU1(GRUBase):

    def __init__(self, input_size=7, hidden_size=16, gru_hidden=8):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
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

    def forward(self, state_batch, hidden):
        x = self.input_proj(state_batch)
        new_hidden = self.gru(x, hidden)
        logits = self.head(new_hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, new_hidden    

    def _init_weights(self):
        '''
        Earlier the model flapped continously in the beginning and slowly learned that it is better to not flap sometimes
        To learn that behaviour it spent 50-60 steps with score 0.
        This was happening as inital prob was >0 on an avg due to intial weight allocation hence it is solved by initializing bias to be 0
        This brings the output to mean 0.5 for no bias in the start of training
        '''
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module,(nn.GRUCell)):
                nn.init.xavier_uniform_(module.weight_hh)
        nn.init.constant_(self.head[-1].bias, -1)



class GRU_no_dropout(GRUBase):

    def __init__(self, input_size=7, hidden_size=16, gru_hidden=8):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_size, gru_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Linear(gru_hidden // 2, 1),
        )
        self._init_weights()

    def forward(self, state_batch, hidden):
        x = self.input_proj(state_batch)
        new_hidden = self.gru(x, hidden)
        logits = self.head(new_hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs, new_hidden    

    def _init_weights(self):
        '''
        Earlier the model flapped continously in the beginning and slowly learned that it is better to not flap sometimes
        To learn that behaviour it spent 50-60 steps with score 0.
        This was happening as inital prob was >0 on an avg due to intial weight allocation hence it is solved by initializing bias to be 0
        This brings the output to mean 0.5 for no bias in the start of training
        '''
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            if isinstance(module,(nn.GRUCell)):
                nn.init.xavier_uniform_(module.weight_hh)
        nn.init.constant_(self.head[-1].bias, -1)