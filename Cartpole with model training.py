from multiprocessing.dummy import DummyProcess
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


env_name = 'CartPole-v1'
env = gym.make(env_name) ## these lines set up a cartpole environment where the the model can take place
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
