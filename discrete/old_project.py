import gym
import numpy as np
from numpy import sin, cos, pi
from tqdm import tqdm
import multiprocessing as mp
import ctypes
import time

from mesh import UniformMesh
from ctg import CTGPool
from transitions import TransitionPool

env = gym.make("gym_underactuated:CustomAcrobot-v0")

resolution = 2*25
err_tol = .001
retrain = True
block_size = 512 * 8

# create state space
LOWERS = np.array([   0,    0, -env.MAX_VEL_1, -env.MAX_VEL_2])
UPPERS = np.array([2*pi, 2*pi,  env.MAX_VEL_1,  env.MAX_VEL_2])
STEPS  = np.array([resolution]*4)

state_space = UniformMesh(LOWERS, UPPERS, STEPS, data_dims=2)

# create action space
action_space = np.linspace(-env.TORQUE_LIMIT, env.TORQUE_LIMIT, resolution)

# l_inf error for convergence
def err_fn(old_ctg, new_ctg):
    return np.max(np.abs(old_ctg - new_ctg))

# min time cost
def cost_fn(state):
    state = np.copy(state)
    state[0] -= pi
    #state[1] -= pi
    if state.dot(state) < .001:
        return 0
    else:
        return 1

err = 1.

def shared_double_array(shape, lock=False):
    """
    Form a shared memory numpy array of doubles.
    
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
    https://stackoverflow.com/questions/38770681/sharing-a-ctypes-numpy-array-without-lock-when-using-multiprocessing
    """

    size = 1
    for dim in shape:
        size *= dim
    
    shared_array = mp.Array(ctypes.c_double, int(size), lock=lock)
    return shared_array

    #shared_array = np.ctypeslib.as_array(shared_array_base)
    #shared_array = shared_array.reshape(*shape)
    #return shared_array

# precompute transition matrix
shape = \
    np.append(
        np.array(state_space.data[0].shape), 
        np.append(
            np.array(action_space.shape), 
            np.array([len(LOWERS)])
        )
    )

shared_array = shared_double_array(shape)
np_arr = np.ctypeslib.as_array(shared_array)
np_arr.fill(np.inf)
transition_matrix = np_arr.reshape(shape)

#from ctg import CTGProcess
#from multiprocessing import Queue
#queue = Queue()
#proc1 = CTGProcess(queue, None, np_arr)
#proc2 = CTGProcess(queue, None, np_arr)
#queue.put(1)
#queue.put(2)
#proc1.start()
#proc1.join()
#print(f"observed {np_arr[0]}")
#proc2.start()
#proc2.join()
#print(f"observed {np_arr[0]}")
#exit()

shared_points = shared_double_array(state_space.points.shape)
np_points = np.ctypeslib.as_array(shared_points).reshape(state_space.points.shape)
np.copyto(np_points, state_space.points)

def precompute_transitions(args):
    start, block_size, points_shape, env, in_shape, out_shape = args
    np_arr = np.ctypeslib.as_array(shared_array)
    np_arr = np_arr.reshape(*out_shape) 

    np_points = np.ctypeslib.as_array(shared_points).reshape(points_shape)
    end = min(start + block_size, len(np_points))

    for idx, state in enumerate(np_points[start:end]):
        idx += start
        idx = np.unravel_index(idx, in_shape)
        end_states = np.array([env.step(action, state)[0] for action in action_space])
        #end_states[(end_states < LOWERS).any(axis=1)] = LOWERS
        #end_states[(end_states > UPPERS).any(axis=1)] = UPPERS
        np_arr[idx] = end_states

#def one_step_ctg(args):
#    start, block_size, state_space, transition_matrix = args
#    end = min(start + block_size, len(state_space.points))
#
#    new_ctg = np.zeros(state_space.data[0].shape)
#
#    for idx, state in enumerate(state_space.points[start:end]):
#        idx += start
#        idx = np.unravel_index(idx, new_ctg.shape)
#
#        cost = cost_fn(state)
#        if cost == 0:
#            # terminal
#            new_ctg[idx] = 0
#            continue
#        #end_states = np.array([env.step(action, state)[0] for action in action_space])
#        end_states = transition_matrix[idx]
#        valid_states = end_states[np.logical_and((end_states >= LOWERS).all(axis=1), (end_states <= UPPERS).all(axis=1))]
#        if len(valid_states) == 0:
#            continue
#        costs_to_go = state_space.interpolate(valid_states)
#        a_idx = np.argmin(costs_to_go)
#        new_ctg[idx] = costs_to_go[a_idx] + 1 # min time cost
#    return new_ctg

#args = [(i, state_space.points[i:min(i+block_size,len(state_space.points))], env, state_space.data[0].shape, shape) \
#        for i in range(0, len(state_space.points), block_size)]
NUM_PROCS = mp.cpu_count() - 2 or 1
#NUM_PROCS = 1
pool = None

try:
    assert not retrain
    transition_matrix = np.load(f"outputs/transition_{resolution}.npy").reshape(shape)
    np_arr = np_arr.reshape(shape)
    np.copyto(np_arr, transition_matrix)
    del transition_matrix
    transition_matrix = np_arr
    print(f"loaded precomputed transition matrices...")
except:
    #args = [(i, block_size, state_space.points.shape, env, state_space.data[0].shape, shape) \
    #        for i in range(0, len(state_space.points), block_size)]
    #start = time.time()
    #if pool is None:
    #    pool = mp.Pool(NUM_PROCS)
    #res = pool.map(precompute_transitions, args)
    #end = time.time()
    #old = np.copy(transition_matrix)
    #print(f"done precomputing transition matrices... took {end - start} seconds")
    #pool.close()

    args = range(0, len(state_space.points), block_size)
    print(f"num tasks: {len(args)} | num procs: {NUM_PROCS}")

    start = time.time()
    pool = TransitionPool(block_size, NUM_PROCS, transition_matrix, np_points, state_space.data[0].shape, action_space)
    #res = pool.map(precompute_transitions, args)
    pool.run()
    end = time.time()
    print(f"done precomputing transition matrices... took {end - start} seconds")
    pool.close()
    pool = None

    #transition_matrix = np.ctypeslib.as_array(shared_array)
    assert (transition_matrix < np.inf).all()
    #assert (transition_matrix == old).all()
    #transition_matrix = transition_matrix.reshape(shape)

    np.save(f"outputs/transition_{resolution}", transition_matrix)

#transition_matrix_old = np.zeros(shape)
#for idx, state in enumerate(tqdm(state_space.points)):
#    idx = np.unravel_index(idx, state_space.data[0].shape)
#    end_states = np.array([env.step(action, state)[0] for action in action_space])
#    end_states[(end_states < LOWERS).any(axis=1)] = LOWERS
#    end_states[(end_states > UPPERS).any(axis=1)] = UPPERS
#    transition_matrix_old[idx] = end_states
#
#print(np.sum(np.abs(transition_matrix - transition_matrix_old)))
#assert((transition_matrix == transition_matrix_old).all())
#input()
#for i in range(transition_matrix.shape[0]):
#    for j in range(transition_matrix.shape[2]):
#        for k in range(transition_matrix.shape[3]):
#            if np.sum(np.abs(transition_matrix_old[i,j,k] - transition_matrix[i,j,k])) == 0:
#                continue
#            print(f"{i} {j} {k}")
#            print(np.sum(np.abs(transition_matrix_old[i,j,k] - transition_matrix[i,j,k])))
#            print(transition_matrix[i,j,k])
#            print(transition_matrix_old[i,j,k])
#            input()
#transition_matrix = transition_matrix_old

parallel = True
try:
    assert not retrain
    ctg = np.load(f"outputs/ctg_{resolution}.npy")
    print(f"loaded precomputed ctg...")
    state_space.data[0] = ctg
    #if pool is not None:
    #    pool.close()
except:
    #if pool is not None:
    #    pool.close()

    #for dt in [.2, .1, .08, .06, .04, .02]:
    #    env.dt = dt
    if parallel:
        shared_ctg1 = shared_double_array(state_space.data[0].shape)
        np_ctg1 = np.ctypeslib.as_array(shared_ctg1).reshape(state_space.data[0].shape)
        shared_ctg2 = shared_double_array(state_space.data[0].shape)
        np_ctg2 = np.ctypeslib.as_array(shared_ctg2).reshape(state_space.data[0].shape)
        pool = CTGPool(block_size, NUM_PROCS, UPPERS, LOWERS, STEPS, transition_matrix, np_points, np_ctg1, np_ctg2)
        state_space.data[0] = np_ctg2

    for i in tqdm(range(10000)):
        if parallel:
            new_ctg_idx = pool.run_one_iter()
            new_ctg = np_ctg1 if new_ctg_idx == 0 else np_ctg2

            #args = [(i, block_size, state_space, transition_matrix) \
            #        for i in range(0, len(state_space.points), block_size)]

            #res = pool.map(one_step_ctg, args)

            #new_ctg = np.sum(res, axis=0)
            #new_ctg = new_ctg.reshape(state_space.data[0].shape)
            #new_err = err_fn(new_ctg, state_space.data[0])
            #new_ctg = np_ctg1
            new_err = err_fn(np_ctg1, np_ctg2)
        else:
            new_ctg = np.zeros(state_space.data[0].shape)

            # parallel
            for idx, state in enumerate(state_space.points):
                idx = np.unravel_index(idx, new_ctg.shape)
                cost = cost_fn(state)
                if cost == 0:
                    # terminal
                    new_ctg[idx] = 0
                    continue
                #end_states = np.array([env.step(action, state)[0] for action in action_space])
                end_states = transition_matrix[idx]
                valid_states = end_states[np.logical_and((end_states >= LOWERS).all(axis=1), (end_states <= UPPERS).all(axis=1))]
                if len(valid_states) == 0:
                    continue
                costs_to_go = state_space.interpolate(valid_states)
                a_idx = np.argmin(costs_to_go)
                new_ctg[idx] = costs_to_go[a_idx] + 1 # min time cost
            new_err = err_fn(new_ctg, state_space.data[0])

            temp = state_space.data[0]
            state_space.data[0] = new_ctg
            new_ctg = temp

        tqdm.write(str(new_err))
        if np.abs((new_err - err) / err) < .00001 and err < err_tol * 10:
            break
        err = new_err
        if err < err_tol:
            break

    if pool is not None:
        pool.close()

    np.save(f"outputs/ctg_{resolution}", state_space.data[0])

# simulate
import cv2

fps = 50
seconds = 20
env.dt = 1. / fps 

for idx, cur_state in enumerate([(0, 0, 0, 0), (pi, 0, 0, 0), (pi / 2., 0, 0 ,0), (1, 0, 0, 0)]):
    out = cv2.VideoWriter(f'output{idx}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (500, 500), False)
    env.state = cur_state
    for _ in range(fps):
        out.write(np.rot90(env.render()))

    for i in range(fps * seconds):
        if i % max(1,int(fps * env.dt)) == 0:
            end_states = np.array([env.step(action, cur_state)[0] for action in action_space])
            valid = np.logical_and((end_states >= LOWERS).all(axis=1), (end_states <= UPPERS).all(axis=1))
            valid_states = end_states[np.logical_and((end_states >= LOWERS).all(axis=1), (end_states <= UPPERS).all(axis=1))]
            costs_to_go = state_space.interpolate(end_states[valid])
            a_idx = np.argmin(costs_to_go)
            action = action_space[valid][a_idx]

        env.step(action, cur_state)
        #env.step(0, cur_state)
        out.write(np.rot90(env.render()))

        cur_state = env.state

    out.release()


