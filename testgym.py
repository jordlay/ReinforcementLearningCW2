import gym
# import pygame
import distutils.spawn

# env = gym.make("Pendulum-v1")
env = gym.make("BipedalWalker-v3")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
# env.reset()
env.close()