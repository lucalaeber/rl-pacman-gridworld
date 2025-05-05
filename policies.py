from environment import CustomEnv
import numpy as np

env = CustomEnv()

def policy_evaluation(env, policy, theta=1e-5, gamma=0.99, max_iterations=1000):
    """
    Evaluate a given policy by computing the state-value function V(s)
    using the iterative policy evaluation method.

    Args:
        env: The environment with attributes grid_size, n_actions, and P.
        policy: A 3D numpy array of shape (rows, cols, n_actions) where
                policy[y, x, a] is the probability of taking action a in state (y, x).
        theta: Convergence threshold.
        gamma: Discount factor.

    Returns:
        V: A 2D numpy array of state values.
    """
    rows, cols = env.grid_size
    V = np.zeros((rows, cols))
    
    for i in range(max_iterations):
        delta = 0
        # Loop over all states
        for y in range(rows):
            for x in range(cols):
                s = (y, x)
                v = V[y, x]
                new_v = 0.0
                # Loop over all actions for state s
                for a in range(env.action_space.n):
                    action_prob = policy[y, x, a]
                    # For each possible outcome for taking action a in state s:
                    for prob, next_state, reward, done in env.P[s][a]:
                        # Assuming next_state is a tuple (y_next, x_next)
                        if done:
                            new_v += action_prob * prob * reward
                        else:
                            new_v += action_prob * prob * (reward + gamma * V[next_state])
                V[y, x] = new_v
                delta = max(delta, abs(v - new_v))
        if delta < theta:
            print(f"Policy evaluation converged after {i+1} iterations.")
            break
        if i == max_iterations - 1:
            print("Policy evaluation reached maximum iterations without full convergence.")
    return V
                

def policy_improvement(V, env, policy, gamma=0.99):
    """
    Improve the policy based on the current state-value function V.
    Returns a new policy and a boolean indicating if the policy is stable.
    
    Args:
        V: The state-value function (2D numpy array).
        env: The environment with attributes grid_size, n_actions, and P.
        gamma: Discount factor.
    
    Returns:
        new_policy: A 3D numpy array with the improved policy.
        policy_stable: True if the policy did not change; False otherwise.
    """
    rows, cols = env.grid_size
    policy_stable = True
    new_policy = np.copy(policy)  # Copy the current policy for comparison

    for y in range(rows):
        for x in range(cols):
            s = (y, x)
            # Compute the action-values for state s
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    if done:
                        action_values[a] += prob * reward
                    else:
                        action_values[a] += prob * (reward + gamma * V[next_state])
            best_action = np.argmax(action_values)
            old_action = np.argmax(policy[y, x])
            if best_action != old_action:
                policy_stable = False
            new_policy[y, x] = np.eye(env.action_space.n)[best_action]
    return new_policy, policy_stable


def policy_iteration(env, theta=1e-5, gamma=0.99):
    """
    Find the optimal policy using Policy Iteration.
    
    Args:
        env: The environment with attributes grid_size, n_actions, and P.
        theta: Convergence threshold for policy evaluation.
        gamma: Discount factor.
    
    Returns:
        policy: The optimal policy as a 3D numpy array.
        V: The optimal state-value function as a 2D numpy array.
    """
    rows, cols = env.grid_size
    # Start with a uniformly random policy.
    policy = np.ones((rows, cols, env.action_space.n)) / env.action_space.n
    while True:
        V = policy_evaluation(env, policy, theta, gamma)
        
        new_policy, policy_stable = policy_improvement(V, env, policy, gamma)
        if policy_stable:
            break
        policy = new_policy
    return policy, V


def value_iteration(env, theta=1e-5, gamma=0.99):
    """
    Find the optimal policy using Value Iteration.
    
    Args:
        env: The environment with attributes grid_size, n_actions, and P.
        theta: Convergence threshold.
        gamma: Discount factor.
    
    Returns:
        policy: The optimal policy as a 3D numpy array.
        V: The optimal state-value function as a 2D numpy array.
    """
    rows, cols = env.grid_size
    V = np.zeros((rows, cols))
    
    while True:
        delta = 0
        for y in range(rows):
            for x in range(cols):
                s = (y, x)
                v = V[y, x]
                action_values = np.zeros(env.action_space.n)
                for a in range(env.action_space.n):
                    for prob, next_state, reward, done in env.P[s][a]:
                        action_values[a] += prob * (reward + gamma * V[next_state])
                V[y, x] = np.max(action_values)
                delta = max(delta, abs(v - V[y, x]))
        if delta < theta:
            break

    # Derive policy from the optimal value function
    policy = np.zeros((rows, cols, env.action_space.n))
    for y in range(rows):
        for x in range(cols):
            s = (y, x)
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            best_action = np.argmax(action_values)
            policy[y, x] = np.eye(env.action_space.n)[best_action]

    return policy, V

