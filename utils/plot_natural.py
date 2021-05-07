import gym
import numpy as np
from numpy import pi

from utils.sim import sim

def plot_acrobot():
    env = gym.make("gym_underactuated:CustomAcrobot-v0")
    procs = []
    for idx, start_state in enumerate(env.sim_states):
        p = sim(f"outputs/videos/natural_acrobot_{idx}", start_state, env)
        procs.append(p)
    [p.join() for p in procs]

def plot_pendulum():
    env = gym.make("gym_underactuated:CustomPendulum-v0")
    procs = []
    for idx, start_state in enumerate(env.sim_states):
        p = sim(f"outputs/videos/natural_pendulum_{idx}", start_state, env)
        procs.append(p)
    [p.join() for p in procs]

if __name__ == "__main__":
    #plot_pendulum()
    plot_acrobot()

