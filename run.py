import numpy as np
import matplotlib.pyplot as plt
import pygame

from environment import CustomEnv
from MC import mc_prediction, mc_control
from TD import td0_prediction, sarsa, q_learning
from policies import policy_iteration, value_iteration


def plot_value_function(V, title="State Value Function", grid_size=(10, 10)):
    arr = np.zeros(grid_size)
    for (y, x), value in V.items():
        arr[y, x] = value
    plt.figure()
    plt.imshow(arr, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_policy_map(policy_dict, title="Policy Map", grid_size=(10, 10)):
    arr = np.zeros(grid_size)
    for (y, x), probs in policy_dict.items():
        arr[y, x] = np.argmax(probs)
    plt.figure()
    plt.imshow(arr, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_reward_evolution(rewards, title="Reward Evolution"):
    episodes = np.arange(len(rewards))
    plt.figure()
    plt.plot(episodes, rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(title)
    plt.show()

def run_policy_on_env(env, policy_dict, n_episodes=1, render=True, delay=200):
    rewards_all = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tuple = (int(state[0]), int(state[1]))
            action = np.argmax(policy_dict.get(state_tuple, np.ones(env.action_space.n) / env.action_space.n))
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                pygame.time.wait(delay)
        rewards_all.append(total_reward)
    return rewards_all

# ---------- Main Combined Execution ----------
def main():
    gamma = 0.99
    theta = 1e-4
    alpha = 0.1
    epsilon = 0.1
    episodes_mc = 3000
    episodes_td = 1000

    run_dp = True
    run_mc = True
    run_td = True
    
    if run_dp:
        print("\n--- Policy Iteration ---")
        env = CustomEnv()
        policy_pi, V_pi = policy_iteration(env, theta=theta, gamma=gamma)
        plot_value_function({(y, x): V_pi[y, x] for y in range(env.grid_size[0]) for x in range(env.grid_size[1])},
                            "Policy Iteration: Value Function", env.grid_size)
        policy_dict = {(y, x): policy_pi[y, x] for y in range(env.grid_size[0]) for x in range(env.grid_size[1])}
        plot_policy_map(policy_dict, "Policy Iteration: Optimal Policy", env.grid_size)
        rewards = run_policy_on_env(env, policy_dict)
        plot_reward_evolution(rewards, "Reward (Policy Iteration)")
        env.close()

    if run_mc:
        env = CustomEnv()
        policy_random = {s: np.ones(env.action_space.n) / env.action_space.n for s in env.P}
        
        print("\n--- Monte Carlo Prediction ---")
        V_mc = mc_prediction(env, policy_random, gamma=gamma, episodes=episodes_mc)
        plot_value_function(V_mc, "MC Prediction: Value Function", env.grid_size)

        print("\n--- Monte Carlo Control ---")
        policy_mc, Q_mc = mc_control(env, gamma=gamma, epsilon=epsilon, episodes=episodes_mc)
        
        rewards = run_policy_on_env(env, policy_mc, render = True)
        
        plot_policy_map(policy_mc, "MC Control: Optimal Policy", env.grid_size)
        plot_reward_evolution(rewards, "Reward (MC Control)")
        env.close()

    if run_td:
        print("\n--- Q-learning ---")
        env = CustomEnv()
        Q, policy_q, rewards_q = q_learning(env, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes_td)
        V_q = {s: max(Q.get((s, a), -np.inf) for a in range(env.action_space.n)) for s in policy_q.keys()}
        plot_value_function(V_q, "Q-learning: Value Function", env.grid_size)
        plot_policy_map(policy_q, "Q-learning: Policy", env.grid_size)
        rewards = run_policy_on_env(env, policy_q)
        plot_reward_evolution(rewards, "Reward (Q-learning)")
        env.close()

    

if __name__ == "__main__":
    main()
