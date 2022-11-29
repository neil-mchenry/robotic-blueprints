import sys
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
import math as math

import warnings
warnings.filterwarnings("ignore")

import gym_race

LR = 1e-3
env = gym.make('Pyrace-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000
threshold = 1000



def some_random_games_first():
    for episode in range(5):
        env.reset()
        score = 0
        observation = env.reset()
        state_0 = state_to_bucket(observation)
        total_reward = 0
        if episode >= threshold:
            explore_rate = 0.01
        for t in range(goal_steps):
            action = select_action(state_0, explore_rate)
            observation, reward, done, info = env.step(action)
            state = state_to_bucket(observation)
            score += reward
            env.render()
            if done:
                print('Episode:', episode, 'Score:', score)
                break

#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                
                elif data[1] == 0:
                    output = [1,0]
                
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted score: ', mean(accepted_scores))
    print('Median acepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

#initial_population()
def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model


training_data = initial_population()
model = train_model(training_data)

scores = [] 
choices = []
env = env.render()
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


if __name__ == "__main__":

    env = gym.make("Pyrace-v0")
    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print(DECAY_FACTOR)

    NUM_EPISODES = 9999999
    MAX_T = 2000
    #MAX_T = np.prod(NUM_BUCKETS, dtype=int) * 100

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    some_random_games_first()
    #load_and_play()

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

