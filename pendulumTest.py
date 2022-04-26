import numpy as np 
import gym
# create and initialize the environment
env = gym.make("Pendulum-v1")
env.reset()
# next_State, reward, done ,info = env.step(action)
class RandomPolicy:
    def __init__(self, env):
        act_dim = []
        self._max_action = self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._act_dim = act_dim
        self.policy_name = "RandomPolicy"
    def get_action(self, state):
            return np.random.uniform(
                low=-self._max_action,
                high=self._max_action,
                size=self._act_dim)
class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like a number of steps and total reward of the last espisode.
    '''
    def __init__(self, env,max_rollout_length = 1000):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self._max_rollout_length = max_rollout_length
        self.total_rew = 0
        self.len_episode = 0
        
    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.total_rew += reward
        self.len_episode += 1
    
    @property
    def get_episode_reward(self):
        return self.total_rew
    @property
    def get_episode_length(self):
        return self.len_episode
policy = RandomPolicy(env)
# play 4 games
number_episodes = 4
number_moves    = 100
for i in range(number_episodes):
    # initialize the environment
    state = env.reset()
    done = False
    game_rew = 0  # accumulated reward
    for j in range(number_moves):
        # choose a random action
        action = policy.get_action(state)
        # take a step in the environment
        next_state, rew, done, info = env.step(action)
        state = next_state
        game_rew += rew
        env.render()
        # when is done, print the cumulative reward of the game and reset the environment
        if done:
            print("Done")
            break
    print('Episode %d finished, reward:%d, the lenght of the episode:%d'% (i, game_rew,j))
# close the redering window
env.close()

# create and the environment
env = gym.make("Pendulum-v1")
env = StructEnv(env)