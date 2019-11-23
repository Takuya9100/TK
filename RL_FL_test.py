import gym
import numpy as np

env = gym.make('FrozenLake-v0')
a = 0.9
discount_factor = 0.8

rewards = []
Q = np.zeros((env.observation_space.n,env.action_space.n))
# Q = np.load("Q.npy")
for i in range(4000):
    state = env.reset()
    episode_reward = 0
    while True:
        noise = np.random.random((1, env.action_space.n)) / i**2.
        action = np.argmax(Q[state,:] + noise)
        state2, reward, done, _ = env.step(action)
        env.render()
        episode_reward += reward
        Qtarget = reward + discount_factor * np.max(Q[state2,:])
        Q[state, action] = (1-a) * Q[state, action] + a * Qtarget
        state = state2
        if done:
            rewards.append(episode_reward)
            break
print(max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]))
