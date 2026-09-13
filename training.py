import numpy as np
from gym import FlappyBirdEnv
from actor import TreePolicy, get_actions


def compute_returns(rewards, gamma=0.99):
    """Discounted return at each timestep, computed backward through the trajectory."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def _setup(population_size, render, load_model: bool = False, model_path: str = None):
    env = FlappyBirdEnv(population_size=population_size, render=render)
    policy = TreePolicy()
    if load_model:
        print(f"Loading pretrained policy from {model_path}....")
        try:
            policy.load(model_path)
            print(f"Loaded pretrained policy from {model_path}")
        except Exception as e:
            print(f"No pretrained policy found ({e}), starting from scratch.")
    return env, policy


def _setup_episode(population_size):
    done = False
    features_per_bird = [[] for _ in range(population_size)]
    actions_per_bird = [[] for _ in range(population_size)]
    rewards_per_bird = [[] for _ in range(population_size)]
    alive_before = np.ones(population_size, dtype=bool)
    prev_scores = np.zeros(population_size, dtype=np.int32)
    prev_action = np.zeros(population_size, dtype=np.float32)
    return done, features_per_bird, actions_per_bird, rewards_per_bird, alive_before, prev_scores, prev_action


def train(num_episodes=500, population_size=100, gamma=0.98, render=False,
          load_model: bool = False, model_path: str = None, save_path: str = None,
          num_boost_round=20, epsilon_start=0.15, epsilon_end=0.02, epsilon_decay_episodes=300):

    env, policy = _setup(population_size, render, load_model, model_path)
    best_score = 10.0

    for episode in range(num_episodes):
        # Linearly anneal the random-exploration floor: high early on so the
        # model can't lock itself into a bad habit before it has any data,
        # lower later once it has something worth exploiting.
        decay_frac = min(1.0, episode / max(1, epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay_frac

        state = env.reset()
        (done, features_per_bird, actions_per_bird, rewards_per_bird,
         alive_before, prev_scores, prev_action) = _setup_episode(population_size)

        while not done:
            actions, probs, features = get_actions(policy, state, prev_action, stochastic=True, epsilon=epsilon)
            next_state, alive_after, scores, done = env.step(actions)
            prev_action = actions.astype(np.float32)

            score_delta = scores - prev_scores
            prev_scores = scores.copy()
            step_reward = 0.1 + (10 * score_delta)

            for i in range(population_size):
                if alive_before[i]:  # Only record steps for birds that were alive this step
                    features_per_bird[i].append(features[i])
                    actions_per_bird[i].append(actions[i])
                    rewards_per_bird[i].append(step_reward[i])

            alive_before = alive_after
            state = next_state

        # --- Build reward-weighted classification targets across all birds' trajectories ---
        all_features, all_targets, all_returns = [], [], []

        for i in range(population_size):
            if len(rewards_per_bird[i]) == 0:
                continue
            returns_to_go = compute_returns(rewards_per_bird[i], gamma=gamma)
            all_features.extend(features_per_bird[i])
            all_targets.extend(actions_per_bird[i])
            all_returns.extend(returns_to_go)

        all_features = np.array(all_features, dtype=np.float32)
        all_targets = np.array(all_targets, dtype=np.float32)
        advantages = np.array(all_returns, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        '''
        Trees can't be nudged by a signed gradient like -log_prob * advantage
        the way the GRU was. Instead: for positive-advantage steps, the label
        is the action actually taken (reinforce it); for negative-advantage
        steps, the label is flipped to the OTHER action (push away from it).
        sample_weight = |advantage| controls how hard each row is reinforced.
        This is a heuristic stand-in for REINFORCE, not an equivalent — if it
        doesn't learn well, an evolutionary/genetic approach on the tree
        ensembles is the more standard way to train non-differentiable
        policies and pairs naturally with a population like this one.
        '''
        targets = np.where(advantages >= 0, all_targets, 1 - all_targets)
        weights = np.abs(advantages) + 1e-3  # avoid rows with ~zero weight

        policy.fit_increment(all_features, targets, weights, num_boost_round=num_boost_round)

        mean_score = float(np.mean(prev_scores))
        max_score = int(np.max(prev_scores))

        if mean_score > best_score:
            best_score = mean_score
            policy.save(save_path)

        print(f"Episode {episode:4d} | mean_score={mean_score:6.2f} | max_score={max_score:3d} | n_samples={len(targets)}")

    env.close()
    return policy


if __name__ == "__main__":
    load_model_path = None
    save_path = r"try_gru/lgbm_2pipe.txt"
    config = {
        "num_episodes": 800,
        "population_size": 200,
        "gamma": 0.97,
        "render": 1,
        "load_model": 0,
        "model_path": save_path,
    }
    trained_policy = train(
        num_episodes=config["num_episodes"],
        population_size=config["population_size"],
        gamma=config["gamma"],
        render=config["render"],
        load_model=config["load_model"],
        model_path=load_model_path,
        save_path=config["model_path"],
    )
    print(f"Saved trained policy to {save_path}")