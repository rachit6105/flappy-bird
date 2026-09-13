import numpy as np
import lightgbm as lgb

import configs


class TreePolicy:
    """
    Tree-based policy replacing the GRU network.

    Important difference from the GRU version: trees have no hidden state to
    carry across timesteps, so there's no `hidden` argument anywhere here.
    The model only ever sees the current step's features (state + prev_action).
    """

    def __init__(self, num_leaves=15, learning_rate=0.05, feature_names=None):
        self.params = {
            "objective": "binary",
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }
        self.booster = None  # lgb.Booster, stays None until the first fit
        self.feature_names = feature_names or [
            "bird_y", "velocity", "pipe1_dx", "pipe1_dy", "pipe2_dx", "pipe2_dy", "prev_action"
        ]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """features: (N, 7) -> probs: (N,)"""
        if self.booster is None:
            # Uninitialized model: bias flap probability low so birds don't
            # flap-spam before any training has happened (same rationale as
            # the GRU's head-bias=-1 init).
            return np.full(features.shape[0], 0.12, dtype=np.float32)
        return self.booster.predict(features)

    def fit_increment(self, X, targets, sample_weight, num_boost_round=20):
        """
        Continue training the booster on one episode's worth of data.
        `targets`/`sample_weight` come from the reward-weighted-classification
        trick in train.py, since boosted trees have no gradient to backprop
        log-probs through the way the GRU did.
        """
        train_set = lgb.Dataset(
            X, label=targets, weight=sample_weight, feature_name=self.feature_names
        )
        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            init_model=self.booster,  # continue boosting on top of previous rounds
            keep_training_booster=True,
        )

    def save(self, path):
        if self.booster is not None:
            self.booster.save_model(path)

    def load(self, path):
        self.booster = lgb.Booster(model_file=path)


def normalize_state(state_matrix):
    normalized = state_matrix.copy().astype(np.float32)
    normalized[:, 0] /= configs.SCREEN_HEIGHT   # bird_y
    normalized[:, 1] /= 10.0                     # velocity
    normalized[:, 2] /= configs.SCREEN_WIDTH     # pipe1_dx
    normalized[:, 3] /= configs.SCREEN_HEIGHT    # pipe1_dy
    normalized[:, 4] /= configs.SCREEN_WIDTH     # pipe2_dx
    normalized[:, 5] /= configs.SCREEN_HEIGHT    # pipe2_dy
    return normalized


def get_actions(policy: TreePolicy, state_matrix, prev_action, stochastic=True,
                 threshold=0.5, prob_floor=0.03, epsilon=0.0):
    """
    No hidden state to thread through anymore, so the signature is shorter
    than the GRU version's get_actions.

    prob_floor: clamps predicted probability into [prob_floor, 1-prob_floor].
        Boosted trees can saturate to near-0/near-1 once training data agrees
        strongly on a leaf; without a floor the minority action can stop
        getting sampled (and therefore stop getting any training data),
        which is a self-reinforcing bias the GRU avoided via clamp+entropy.
    epsilon: probability of ignoring the model entirely and picking a
        uniformly random action instead. This is a harder exploration floor
        than prob_floor -- it doesn't depend on the tree being well-calibrated
        at all, it just guarantees a trickle of data on both actions.

    Returns: (actions, probs, features)
        - probs are raw probabilities (not log_probs) since we need them for
          the weighted-classification update, not autograd.
        - features are returned so train.py can log exactly what was fed to
          the model at this step, for the next fit_increment() call.
    """
    normalized = normalize_state(state_matrix)
    features = np.concatenate([normalized, prev_action.reshape(-1, 1)], axis=1).astype(np.float32)
    probs = policy.predict_proba(features)
    probs = np.clip(probs, prob_floor, 1.0 - prob_floor)

    if stochastic:
        actions = (np.random.rand(*probs.shape) < probs).astype(int)
        if epsilon > 0.0:
            random_mask = np.random.rand(*probs.shape) < epsilon
            random_actions = np.random.randint(0, 2, size=probs.shape)
            actions = np.where(random_mask, random_actions, actions)
    else:
        actions = (probs > threshold).astype(int)
    return actions, probs, features