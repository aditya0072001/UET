import gym
import numpy as np

# Initialize the environment
env = gym.make('CartPole-v1')

# Initialize the Q-table
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# Set the learning rate and discount factor
lr = 0.01
gamma = 0.99

# Initialize the epsilon value
epsilon = 1.0

# Loop over episodes
for episode in range(1000):

    # Initialize the state
    state = env.reset()

    # Loop over steps in the episode
    for _ in range(100):

        # Choose an action based on epsilon-greedy policy
        action = np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(0, action_size)

        print(env.step(action))

        # Perform the action and observe the reward and next state
        next_state, reward, done = env.step(action)[0:3]
        next_state = np.reshape(next_state, (state_size, 1))

        print('Episode:', episode, 'Step:', _, 'State:', state, 'Action:', action, 'Reward:', reward, 'Next state:', next_state, 'Done:', done)

        # Update the Q-table
        Q[state, action] += lr * (reward + gamma * np.max(Q[next_state, :], axis=None))

        # Update the state
        state = next_state

    # Decrease epsilon
    epsilon *= 0.99

# Evaluate the learned policy
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    for _ in range(100):
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)[0:3]
        total_reward += reward
        print(total_reward)
        if done:
            break
    test_rewards.append(total_reward)

# Print the average rewards over test episodes
print('Average rewards over test episodes:', np.mean(test_rewards))

# plot the convergence of the Q-learning algorithm

plt.figure(figsize=(12, 5))
plt.plot(Q.sumRewardsEpisode, color='blue', linewidth = 1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()

# Plot the histogram of rewards from random  strategy
plt.hist(total_reward)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# close the environment
env.close()


