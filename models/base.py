import torch
import numpy as np
import game.configs as configs
from pathlib import Path
import torch.nn as nn

from models.registry import UserError, get_model


def normalize_state(state_matrix):
    normalized = np.array(state_matrix, dtype=np.float32)
    divisors = np.array([
        configs.SCREEN_HEIGHT, # 0: bird_y
        10.0,                  # 1: velocity
        configs.SCREEN_WIDTH,  # 2: pipe1_dx
        configs.SCREEN_HEIGHT, # 3: pipe1_dy
        configs.SCREEN_WIDTH,  # 4: pipe2_dx
        configs.SCREEN_HEIGHT, # 5: pipe2_dy
        1.0
    ], dtype=np.float32)
    
    return normalized / divisors


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



class Policy(nn.Module):
    """Base for every policy family. Provides self-describing save/load."""

    uses_history = False  # @register_model(history=True) overrides this

    @property
    def history(self) -> int:
        """Number of stacked frames the model expects (0 = single-frame)."""
        return self.init_kwargs.get("K", 1) if self.uses_history else 0

    def save(self, path, metrics=None):
        if not hasattr(self, "registry_name"):
            raise TypeError(f"{type(self).__name__} isn't registered; add @register_model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.registry_name,
                "kwargs": self.init_kwargs,
                "state_dict": self.state_dict(),
                "metrics": {k: float(v) for k, v in (metrics or {}).items()},
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None, **legacy_kwargs):
        """Build a model from a checkpoint and return it in eval mode.

        `Policy.load(p)` works for any model, `GRUBase.load(p)` accepts any GRU,
        and `GRU1.load(p)` insists on GRU1. Legacy checkpoints (a bare
        state_dict) need a concrete class plus `legacy_kwargs`.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(path)
        if not path.is_file():
            raise UserError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=True)

        if isinstance(ckpt, dict) and {"model", "kwargs", "state_dict"} <= ckpt.keys():
            model_cls = get_model(ckpt["model"])
            if not issubclass(model_cls, cls):
                raise UserError(f"{path} holds {ckpt['model']}, which is not a {cls.__name__}.")
            model = model_cls(**ckpt["kwargs"])
            state, metrics = ckpt["state_dict"], ckpt.get("metrics", {})
        else:
            if not hasattr(cls, "registry_name"):
                raise UserError(
                    f"{path} is a legacy checkpoint (bare state_dict); "
                    f"give the full model name, e.g. `gru.GRU1`."
                )
            model, state, metrics = cls(**legacy_kwargs), ckpt, {}

        model.load_state_dict(state)
        model.metrics = metrics
        return model.to(device)