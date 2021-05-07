import gym
from multiprocessing import Process, Queue
import numpy as np
from numpy import pi
import time
from tqdm import tqdm

from mesh import UniformMesh

def precompute_transitions(start, block_size, transition_matrix, points, shape, env, action_space):
    end = min(start + block_size, len(points))
    for idx, state in enumerate(points[start:end]):
        idx += start
        idx = np.unravel_index(idx, shape)
        end_states = np.array([env.step(action, state)[0] for action in action_space])
        transition_matrix[idx] = end_states

class TransitionProcess(Process):
    def __init__(self, idx, queue, transition_matrix, points, shape, action_space):
        super(TransitionProcess, self).__init__()
        self.idx = idx

        self.queue = queue 

        self.transition_matrix = transition_matrix
        self.points = points
        self.shape = shape
        self.action_space = action_space

        self.env = gym.make("gym_underactuated:CustomAcrobot-v0")

    def run(self):
        try:
            while True:
                args = self.queue.get()
                if args is None:
                    return
                else:
                    start, block_size = args
                    precompute_transitions(start, block_size, self.transition_matrix, 
                                           self.points, self.shape, self.env, self.action_space)
        except KeyboardInterrupt:
            return

class TransitionPool(object):
    def __init__(self, block_size, num_procs, transition_matrix, points, shape, action_space):
        self.block_size = block_size
        self.num_procs = num_procs

        self.transition_matrix = transition_matrix
        self.points = points

        self.shape = shape

        self.action_space = action_space

        self.queue = Queue()

        self.pool = [self.make_proc(idx) for idx in range(self.num_procs)]
        [proc.start() for proc in self.pool]

    def make_proc(self, idx):
        return TransitionProcess(idx,
                                 self.queue, 
                                 self.transition_matrix, 
                                 self.points, 
                                 self.shape,
                                 self.action_space)

    def close(self):
        [proc.terminate() for proc in self.pool]
        self.pool = None

    def run(self):
        assert self.pool is not None
        assert self.queue.empty()
        args = [(i, self.block_size) for i in range(0, len(self.points), self.block_size)]

        args.extend([None] * len(self.pool))
        [self.queue.put(arg) for arg in args]

        total = len(args) - len(self.pool)
        with tqdm(total=total) as pbar:
            while True:
                time.sleep(0.1)
                cur_size = min(total, max(0, self.queue.qsize()))
                pbar.update(total - cur_size)
                total = cur_size
                if total == 0:
                    break
        [proc.join() for proc in self.pool]

