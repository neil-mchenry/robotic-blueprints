from curses import wrapper
import gym
from gym.wrappers import Monitor
from stable_baselines3 import ppo
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


## these lines set up a cartpole environment where the the model can take place
env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, 'C:\\Users\\mluke\\Desktop\\recordings' , video_callable=lambda episode_id: True)
for episode in range(1,11):
    score = 0
    state = env.reset()
    done = False

    while not done:
        env.render()
        action =env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward

    print('Episode:', episode, 'Score:', score)
env.close()


