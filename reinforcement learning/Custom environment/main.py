import numpy as np
import sys
import numpy as np
import math
import random
import gym
import gym_game
def simulate():
    ---

if __name__ =="__main__":
    env = gym.make(Pygame-v0)
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()