from queue import Empty 
from multiprocessing import Process, Queue
import numpy as np
from numpy import pi
import time

from mesh import UniformMesh

# min time cost
def cost_fn(state):
    state = np.copy(state)
    state[0] -= pi
    if state.dot(state) < .001:
        return 0
    else:
        return 1

def one_step_ctg(start, block_size, state_space, transition_matrix, new_ctg):
    end = min(start + block_size, len(state_space.points))
    for idx, state in enumerate(state_space.points[start:end]):
        idx += start
        idx = np.unravel_index(idx, new_ctg.shape)

        cost = cost_fn(state)
        if cost == 0:
            # terminal
            new_ctg[idx] = 0
            continue
        end_states = transition_matrix[idx]
        valid_states = end_states[np.logical_and((end_states >= state_space.lowers).all(axis=1), 
                                                 (end_states <= state_space.uppers).all(axis=1))]
        if len(valid_states) == 0:
            continue
        costs_to_go = state_space.interpolate(valid_states)
        new_ctg[idx] = np.min(costs_to_go) + 1 # min time cost

class CTGProcess(Process):
    def __init__(self, idx, work_queue, result_queue, uppers, lowers, steps, transition_matrix, points, ctg1, ctg2):
        super(CTGProcess, self).__init__()
        self.idx = idx

        self.work_queue = work_queue 
        self.result_queue = result_queue

        self.transition_matrix = transition_matrix
        self.points = points
        self.ctg = [ctg1, ctg2]
        self.active = 0

        self.mesh = UniformMesh(lowers, uppers, steps, data_init=[self.ctg[self.active]], points_init=self.points)

    def run(self):
        while True:
            args = self.work_queue.get()
            new_ctg = self.ctg[1 - self.active]
            if args is None:
                self.mesh.data[0] = new_ctg
                self.active = 1 - self.active
                self.result_queue.put(self.idx)
                while self.result_queue.qsize() > 0:
                    time.sleep(.1)
            else:
                #if self.idx == 0:
                #    print(f"start max {np.max(new_ctg)}")
                start, block_size = args
                one_step_ctg(start, block_size, self.mesh, self.transition_matrix, new_ctg)

class CTGPool(object):
    def __init__(self, block_size, num_procs, uppers, lowers, steps, transition_matrix, points, ctg1, ctg2):
        self.block_size = block_size
        self.num_procs = num_procs

        self.uppers = uppers
        self.lowers = lowers
        self.steps = steps

        self.transition_matrix = transition_matrix
        self.points = points
        self.ctg1 = ctg1
        self.ctg2 = ctg2

        self.work_queue = Queue()
        self.result_queue = [Queue() for _ in range(self.num_procs)]

        self.pool = [self.make_proc(idx) for idx in range(self.num_procs)]
        [proc.start() for proc in self.pool]

    def make_proc(self, idx):
        return CTGProcess(idx,
                          self.work_queue, self.result_queue[idx],
                          self.uppers, self.lowers, self.steps, 
                          self.transition_matrix, 
                          self.points, self.ctg1, self.ctg2)

    def close(self):
        [proc.terminate() for proc in self.pool]
        self.pool = None

    def run_one_iter(self):
        assert self.pool is not None
        assert self.work_queue.empty()
        assert [rqueue.empty for rqueue in self.result_queue]
        args = [(i, self.block_size) for i in range(0, len(self.points), self.block_size)]
        args.extend([None] * len(self.pool))
        [self.work_queue.put(arg) for arg in args]
        [rqueue.get() for rqueue in self.result_queue]
        return 1 - self.pool[0].active

        #got = []
        #for i in range(len(self.pool)):
        #    print(f"waiting for {i}")
        #    got.append(self.result_queue.get())
        #    got.sort()
        #    print(got)
        #    print(len(got))

        #[self.result_queue.get() for _ in self.pool]
        #print("what")

