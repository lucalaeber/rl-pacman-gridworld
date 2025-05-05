{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 As part of an advanced AI course at Radboud University, we developed a complete reinforcement learning (RL) framework and custom environment inspired by Pacman. The environment is a 10\'d710 deterministic gridworld where an agent starts in the bottom-left corner and must reach a goal in the top-right corner, while avoiding static monsters and walls that act as obstacles.\
\
Our main objective was to compare and evaluate the performance of key RL algorithm families in terms of policy quality, learning efficiency, and reward optimization. The agent\'92s task was to learn the optimal policy purely from interaction or planning, depending on the method used.\
\
Key components:\
	\'95	Environment Design: Custom gridworld with reward shaping:\
	\'95	-2 for hitting walls, -1 for invalid/empty moves, -5 for encountering monsters, +100 for reaching the goal\
	\'95	Static monsters and randomized but solvable wall layouts\
	\'95	Dynamic Programming (DP): Implemented Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration (using full transition dynamics)\
	\'95	Monte Carlo (MC) Methods: First-Visit Monte Carlo Prediction and Control (on-policy, model-free)\
	\'95	Temporal Difference (TD) Learning: TD(0), SARSA (on-policy), and Q-Learning (off-policy)\
\
Evaluation & Analysis:\
	\'95	Developed state-value heatmaps, policy maps, and reward evolution plots for each algorithm\
	\'95	Experimented with different discount factors (\uc0\u947 ) and convergence thresholds (\u952 )\
	\'95	Compared sample efficiency, policy stability, and convergence behavior across methods\
\
Findings:\
	\'95	SARSA provided the most stable and risk-averse policy in obstacle-heavy environments\
	\'95	Q-learning was fast but exhibited more variance and instability near penalties\
	\'95	Policy Iteration outperformed Value Iteration in cumulative rewards due to faster convergence\
	\'95	Monte Carlo methods were robust in model-free settings but required more episodes to converge\
\
Technologies Used: Python, NumPy, Matplotlib, OpenAI Gym (custom), Jupyter Notebook}