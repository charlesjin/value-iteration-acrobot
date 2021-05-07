import gym
import numpy as np
from numpy import pi
from tqdm import tqdm
import multiprocessing as mp
import time

from mesh import UniformMesh
from ctg import CTGPool
from transitions import TransitionPool
from shmem import shared_double_array
from sim import sim

env = gym.make("gym_underactuated:CustomAcrobot-v0")

resolution = 2*32
err_tol = .01
retrain = False
block_size = resolution ** 4 // 256 #512 * 8

NUM_PROCS = max(mp.cpu_count() - 2, 1)

# create state space
LOWERS = np.array([   0, -pi, -env.MAX_VEL_1, -env.MAX_VEL_2])
UPPERS = np.array([2*pi,  pi,  env.MAX_VEL_1,  env.MAX_VEL_2])
STEPS  = np.array([resolution]*4)
state_space = UniformMesh(LOWERS, UPPERS, STEPS, data_dims=2)

# create action space
action_space = np.linspace(-env.TORQUE_LIMIT, env.TORQUE_LIMIT, resolution)

## create transition matrix
#shape = \
#    np.append(
#        np.array(state_space.data[0].shape), 
#        np.append(
#            np.array(action_space.shape), 
#            np.array([len(LOWERS)])
#        )
#    )
#transition_matrix = shared_double_array(shape)
#transition_matrix.fill(np.inf)
#
## create shared points
#points = shared_double_array(state_space.points.shape)
#np.copyto(points, state_space.points)
#
#try:
#    assert not retrain
#    loaded = np.load(f"outputs/precompute/new_transition_{resolution}.npy").reshape(shape)
#    np.copyto(transition_matrix, loaded)
#    del loaded
#    print(f"loaded precomputed transition matrices...")
#except Exception as e:
#    print(e)
#    args = range(0, len(state_space.points), block_size)
#    print(f"num tasks: {len(args)} | num procs: {NUM_PROCS}")
#
#    start = time.time()
#    pool = TransitionPool(block_size, NUM_PROCS, transition_matrix, points, state_space.data[0].shape, action_space)
#    pool.run()
#    end = time.time()
#    print(f"done precomputing transition matrices... took {end - start} seconds")
#    pool.close()
#
#    assert (transition_matrix < np.inf).all()
#
#    np.save(f"outputs/precompute/new_new_transition_{resolution}", transition_matrix)
#
#try:
#    assert not retrain
#    ctg = np.load(f"outputs/precompute/new_new_ctg_{resolution}.npy")
#    print(f"loaded precomputed ctg...")
#    state_space.data[0] = ctg
#except:
#    # l_inf error for convergence
#    def err_fn(old_ctg, new_ctg):
#        return np.max(np.abs(old_ctg - new_ctg))
#
#    ctg1 = shared_double_array(state_space.data[0].shape)
#    ctg2 = shared_double_array(state_space.data[0].shape)
#
#    pool = CTGPool(block_size, NUM_PROCS, UPPERS, LOWERS, STEPS, transition_matrix, points, ctg1, ctg2)
#    state_space.data[0] = ctg2
#
#    p = None
#    err = 1.
#    pbar = tqdm(range(10000))
#    for i in pbar: 
#        if p is not None:
#            p.join()
#            p = None
#
#        thresh = max(.01, min(50, 50 - (i - 500) * .005))
#        #thresh = .1
#        ctg_idx = pool.run_one_iter(thresh)
#        new_ctg = ctg1 if ctg_idx == 0 else ctg2
#        new_err = err_fn(ctg1, ctg2)
#
#        pbar.set_postfix_str(f"err: {new_err}")
#        if np.abs((new_err - err) / err) < .00001 and err < err_tol * 10:
#            break
#        err = new_err
#        if err < err_tol:
#            break
#
#        if (1+i) % 100 == 0:
#            state_space.data[0] = new_ctg
#            p = sim(f"outputs/videos/output_epoch_{1+i}", (0,0,0,0), env, state_space, action_space, thresh)
#            np.save(f"outputs/precompute/new_new_ctg_{resolution}", new_ctg)
#
#    pool.close()
#
#    state_space.data[0] = new_ctg
#    np.save(f"outputs/precompute/new_new_ctg_{resolution}", state_space.data[0])

# simulate
procs = []
for idx, start_state in enumerate([(0, 0, 0, 0), (pi, 0, 0, 0), (pi / 2., 0, 0 ,0), (1, 0, 0, 0)]):
    #p = sim(f"outputs/videos/output_{idx}", start_state, env, state_space, action_space)
    #procs.append(p)
    p = sim(f"outputs/videos/natural_{idx}", start_state, env)
    procs.append(p)
[p.join() for p in procs]

