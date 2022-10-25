from pdb import Restart
import gym
from gym.envs.classic_control.cartpole import * 
from pyglet.window import key
import matplotlib.pyplot as plt
import time

bool_do_not_quit = True 
scores = []
a = 0

def key_press(k,mod):
    global bool_do_not_quit, a, restart
    if k==0xff0d: restart = True
    if k==key.Escape: bool_do_not_quit = False
    if k==key.Q: bool_do_not_quit = False
    if k==key.LEFT: a = 0 
    if k==key.RIGHT: a = 1

def run_cartPole_asHuman(policy=None, record_video=False):
    env = CartPoleEnv()

    env.reset()
    env.