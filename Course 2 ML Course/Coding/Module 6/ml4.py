import gym
import numpy as np

# Create the environment
env = gym.make('CartPole-v1')

# Initialize Q-values for state-action pairs
Q = {}

# Set hyperparameters
learning_rate = 0.1
discount = 0.9
episodes = 10000
exploration = 0.3

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        state_tuple = state  
        if (state_tuple, 0) not in Q:
            Q[(state_tuple, 0)] = 0
        if (state_tuple, 1) not in Q:
            Q[(state_tuple, 1)] = 0
        if Q[(state_tuple, 0)] > Q[(state_tuple, 1)]:
            action = 0
        else:
            action = 1
        if np.random.random() < exploration:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state_tuple = next_state  # No need to convert, next_state is already a tuple
        if (next_state_tuple, 0) not in Q:
            Q[(next_state_tuple, 0)] = 0
        if (next_state_tuple, 1) not in Q:
            Q[(next_state_tuple, 1)] = 0
        if done:
            Q[(state_tuple, action)] += learning_rate * (reward - Q[(state_tuple, action)])
        else:
            Q[(state_tuple, action)] += learning_rate * (reward + discount * max(Q[(next_state_tuple, 0)], Q[(next_state_tuple, 1)]) - Q[(state_tuple, action)])
        state = next_state

# Evaluate the learned policy
total_reward = 0
num_episodes_eval = 100

for episode in range(num_episodes_eval):
    state = env.reset()
    done = False
    while not done:
        state_tuple = state  # No need to convert, state is already a tuple
        if (state_tuple, 0) not in Q:
            Q[(state_tuple, 0)] = 0
        if (state_tuple, 1) not in Q:
            Q[(state_tuple, 1)] = 0
        if Q[(state_tuple, 0)] > Q[(state_tuple, 1)]:
            action = 0
        else:
            action = 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

print("Average reward per episode: {}".format(total_reward / num_episodes_eval))
