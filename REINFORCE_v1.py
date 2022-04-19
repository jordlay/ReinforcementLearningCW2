# TO DO:
# work out movements, different to racetrack steps 
# look at docs for environment set up
# random policy first or basic NN for simplicity 
# rewards-to-go! entire episode generation
# ADD BASELINE
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
env = gym.make('Pendulum-v1')
# env = gym.make("BipedalWalker-v3")
# Example of Pendulum Running
# for i_episode in range(20):
#     state = env.reset()
#     for t in range(100):
#         env.render()
#         print(state)
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
# define policy network
class policy_net(nn.Module):
    def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(policy_net, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x

class REINFORCEAgent():
    def _init_(self):
        self.step_size = 0.2 # to change
        self.discount_factor = 1 # for simplicity
        self.theta = 0.2 # to change (should be array later)

    def generateEpisode(self):
        episode_rewards = []
        states = []
        actions = []
        env.render()
        done = False
        # print(state)
        # while not done: check this
        for t in range(100):
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            print("S+R", state, reward)
            episode_rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = new_state
            # print(t)
        return episode_rewards, states, actions
    
    def getPolicy(self):
        return True

# # ACTUAL RUN

policy = policy_net(env.observation_space.shape[0], 20, env.action_space.n)

agent = REINFORCEAgent()
state = env.reset()
rewards, states, actions = agent.generateEpisode()

print("rewards for episode", rewards)
sum_rewards = sum(rewards)
print("sum rewards", sum_rewards)
probs = policy(states)
sampler = Categorical(probs)
# for i in range(20):
#     # generate episode
#     # rewards
#     # update policy?
#     # next episode
# print("DONE")
