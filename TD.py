from environment import CustomEnv
import numpy as np
from collections import defaultdict
import random



def td0_prediction(env, policy, gamma=0.99, episodes=5000, alpha=0.1):
    """
    Tabular TD(0) prediction: estimates V(s) under a given policy.
    Args:
        env: The environment.
        policy: A policy.
        gamma: Discount factor.
        episodes: Number of episodes to run.
        alpha: Learning rate (step size).
    Returns:
        V: A dictionary mapping state -> value estimate.
    """
    V = {}
    for s in env.P:
        V[s] = 0.0

    # Loop for each episode
    for i in range(episodes):
        if (i + 1) % (episodes // 10) == 0:
             print(f"TD(0) Prediction: Episode {i+1}/{episodes}")
             
        state_np, _ = env.reset()
        state = tuple(int(x) for x in state_np)
        done = False

        while not done:
            if state not in policy:
                 action = env.action_space.sample()
            else:
                action_probs = policy[state]
                action = np.random.choice(env.action_space.n, p=action_probs)

            # Take action A, observe R, S'
            next_state_np, reward, done, _, _ = env.step(action)
            next_state = tuple(int(x) for x in next_state_np) 
            # V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
            td_target = reward + gamma * V[next_state]
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error
            state = next_state
    return V

def choose_action_epsilon_greedy(state, Q, epsilon, n_actions):
    """
    Chooses an action using an epsilon-greedy policy derived from Q.
    Args:
        state: The current state (tuple).
        Q: The Q-value dictionary Q[state, action].
        epsilon: The probability of choosing a random action.
        n_actions: The number of possible actions.
    Returns:
        action: The chosen action.
    """
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(n_actions))
    else:
        q_values = [Q.get((state, a), 0.0) for a in range(n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

def sarsa(env, gamma=0.99, episodes=5000, alpha=0.1, epsilon=0.1):
    """
    Sarsa (On-Policy TD Control) for estimating Q-values and finding an optimal policy.
    Args:
        env: The environment.
        gamma: Discount factor.
        episodes: Number of episodes to run.
        alpha: Learning rate (step size).
        epsilon: Exploration rate for epsilon-greedy policy.
    Returns:
        Q: A dictionary mapping (state, action) -> Q-value estimate.
        policy: A dictionary mapping state -> action probabilities (derived greedily from Q).
    """
    Q = defaultdict(float)
    n_actions = env.action_space.n
    print(f"Starting Sarsa for {episodes} episodes...")
    print(f"Parameters: gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
    episode_rewards = []
    for i in range(episodes):
        state_np, _ = env.reset()
        state = tuple(int(x) for x in state_np)
        done = False
        total_reward = 0
        step_count = 0
        max_steps_per_episode = env.grid_size[0] * env.grid_size[1] * 5 # Safety break
        action = choose_action_epsilon_greedy(state, Q, epsilon, n_actions)
        while not done and step_count < max_steps_per_episode:
            next_state_np, reward, done, _, _ = env.step(action)
            next_state = tuple(int(x) for x in next_state_np)
            total_reward += reward
            next_action = choose_action_epsilon_greedy(next_state, Q, epsilon, n_actions)
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            td_target = reward + gamma * Q[(next_state, next_action)]
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] = Q[(state, action)] + alpha * td_error
            state = next_state
            action = next_action
            step_count += 1

        episode_rewards.append(total_reward)
        if (i + 1) % max(1, episodes // 20) == 0:
             avg_reward = np.mean(episode_rewards[-100:])
             print(f"  Sarsa Episode {i+1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f}")

    print("Sarsa finished.")
    policy = {}
    all_states = set(s for s, a in Q.keys())

    for s in all_states:
        q_values = [Q.get((s, a), 0.0) for a in range(n_actions)]
        best_action = np.argmax(q_values)
        action_probs = np.zeros(n_actions)
        action_probs[best_action] = 1.0
        policy[s] = action_probs

    return Q, policy, episode_rewards

def q_learning(env, gamma=0.99, episodes=5000, alpha=0.1, epsilon=0.1):
    """
    Q-learning (Off-Policy TD Control) for estimating Q-values and finding an optimal policy.
    Args:
        env: The environment.
        gamma: Discount factor.
        episodes: Number of episodes to run.
        alpha: Learning rate (step size).
        epsilon: Exploration rate for epsilon-greedy policy (behavior policy).
    Returns:
        Q: A dictionary mapping (state, action) -> Q-value estimate.
        policy: A dictionary mapping state -> action probabilities (greedy policy derived from Q).
        episode_rewards: List of cumulative rewards per episode.
    """
    Q = defaultdict(float)
    n_actions = env.action_space.n
    print(f"Starting Q-learning for {episodes} episodes...")
    print(f"Parameters: gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
    episode_rewards = []
    for i in range(episodes):
        state_np, _ = env.reset()
        state = tuple(int(x) for x in state_np)
        done = False
        total_reward = 0
        step_count = 0
        max_steps_per_episode = env.grid_size[0] * env.grid_size[1] * 5 # Safety break
        while not done and step_count < max_steps_per_episode:
            action = choose_action_epsilon_greedy(state, Q, epsilon, n_actions)
            next_state_np, reward, done, _, _ = env.step(action)
            next_state = tuple(int(x) for x in next_state_np)
            total_reward += reward
            if done:
                max_next_q = 0.0
            else:
                next_q_values = [Q.get((next_state, a), 0.0) for a in range(n_actions)]
                max_next_q = max(next_q_values)
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
            td_target = reward + gamma * max_next_q
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] = Q[(state, action)] + alpha * td_error
            state = next_state
            step_count += 1

        episode_rewards.append(total_reward)
        if (i + 1) % max(1, episodes // 20) == 0:
             avg_reward = np.mean(episode_rewards[-100:])
             print(f"  Q-learning Episode {i+1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f}")

    print("Q-learning finished.")
    policy = {}
    all_states = set(s for s, a in Q.keys())
    for s in all_states:
        q_values = [Q.get((s, a), 0.0) for a in range(n_actions)]
        best_action = np.argmax(q_values)
        action_probs = np.zeros(n_actions)
        action_probs[best_action] = 1.0
        policy[s] = action_probs

    return Q, policy, episode_rewards