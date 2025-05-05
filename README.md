# Pacman-Style Gridworld: Reinforcement Learning Project

As part of an advanced AI course at Radboud University, we developed a complete reinforcement learning (RL) framework and custom environment inspired by Pacman. The environment is a 10×10 deterministic gridworld where an agent starts in the bottom-left corner and must reach a goal in the top-right corner, while avoiding static monsters and walls that act as obstacles.

Our main objective was to compare and evaluate the performance of key RL algorithm families in terms of policy quality, learning efficiency, and reward optimization. The agent’s task was to learn the optimal policy purely from interaction or planning, depending on the method used.

## Key Components

- **Environment Design**: Custom gridworld with reward shaping:
  - -2 for hitting walls
  - -1 for invalid/empty moves
  - -5 for encountering monsters
  - +100 for reaching the goal
- **Dynamic Programming (DP)**: Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration
- **Monte Carlo (MC) Methods**: First-Visit Monte Carlo Prediction and Control (on-policy, model-free)
- **Temporal Difference (TD) Learning**: TD(0), SARSA (on-policy), Q-Learning (off-policy)

## Evaluation & Analysis

- Developed state-value heatmaps, policy maps, and reward evolution plots for each algorithm
- Experimented with different discount factors (γ) and convergence thresholds (θ)
- Compared sample efficiency, policy stability, and convergence behavior across methods

## Findings

- **SARSA** provided the most stable and risk-averse policy in obstacle-heavy environments
- **Q-learning** was fast but exhibited more variance and instability near penalties
- **Policy Iteration** outperformed Value Iteration in cumulative rewards due to faster convergence
- **Monte Carlo methods** were robust in model-free settings but required more episodes to converge

## Technologies Used

Python, NumPy, Matplotlib, OpenAI Gym (custom), Jupyter Notebook

## Contributors

- Juan Navarro — s1097545  
- Luca Laber — s1089333  
- Konstantinos Konstantinou — s1118596  
- Levente Bódi — s1122062