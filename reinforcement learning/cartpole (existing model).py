import gym
import random
from curses import wrapper
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow import keras
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")




model = load_model("C://Users\\mluke\\Desktop\\recordings\\tensorflow7\\7model\\model.h5")

goal_steps = 500
score_requirement = 50
initial_games = 10000

scores = [] 
choices = []

env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, 'C:\\Users\\mluke\\Desktop\\recordings\\tens1' , video_callable=lambda episode_id: True)
env.reset()
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)
    print(score)

print('Average Score: ', (sum(scores)) / (len(scores)))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))