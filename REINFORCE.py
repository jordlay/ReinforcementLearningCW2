# import gym
# # import pygame
# import distutils.spawn
# # TO DO:
# # work out movements, different to racetrack steps 
# # look at docs for environment set up
# env = gym.make("Pendulum-v1")
# # env = gym.make("BipedalWalker-v3")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# # env.reset()
# env.close()
import gym
env = gym.make('Pendulum-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()