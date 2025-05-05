from environment import CustomEnv
import numpy as np

def mc_prediction(env, policy, gamma=0.99, episodes=200):
    """
    First-visit MC prediction: estimates V(s) under a given policy.
    """
    V = {}
    returns = {}
    for s in env.P:
        V[s] = 0
        returns[s] = []

    for _ in range(episodes):
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            s = tuple(int(x) for x in state)
            a = np.random.choice(env.action_space.n, p=policy[s])
            next_state, reward, done, _, _ = env.step(a)
            episode.append((s, a, reward))
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, _, r = episode[t]
            G = gamma * G + r
            if s not in visited:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visited.add(s)

    return V

def mc_control(env, gamma=0.99, epsilon=0.1, episodes=200):
    """
    First-visit MC control using epsilon-greedy policy.
    """
    Q = {}
    returns = {}
    for s in env.P:
        Q[s] = np.zeros(env.action_space.n)
        returns[s] = {a: [] for a in range(env.action_space.n)}

    policy = {}
    for s in env.P:
        policy[s] = np.ones(env.action_space.n) / env.action_space.n

    for _ in range(episodes):
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            s = tuple(int(x) for x in state)
            probs = policy[s]
            a = np.random.choice(env.action_space.n, p=probs)
            next_state, reward, done, _, _ = env.step(a)
            episode.append((s, a, reward))
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                returns[s][a].append(G)
                Q[s][a] = np.mean(returns[s][a])
                best_action = np.argmax(Q[s])
                # update epsilon-greedy policy
                policy[s] = np.ones(env.action_space.n) * epsilon / env.action_space.n
                policy[s][best_action] += 1 - epsilon
                visited.add((s, a))

    return policy, Q