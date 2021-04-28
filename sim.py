import cv2
import numpy as np

fps = 50
seconds = 20

def sim(fn, start_state, env, state_space=None, action_space=None):
    out = cv2.VideoWriter(f'{fn}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (500, 500), False)
    env.state = start_state
    env.dt = 1. / fps 

    start_frame = np.rot90(env.render())
    for _ in range(fps // 2):
        out.write(start_frame)

    cur_state = start_state
    for i in range(fps * seconds):
        if state_space is None or action_space is None:
            # simulate the natural dynamics
            env.step(0, cur_state)
        else:
            # simulate the ctg controller
            end_states = np.array([env.step(action, cur_state)[0] for action in action_space])
            valid = np.logical_and((end_states >= state_space.lowers).all(axis=1), (end_states <= state_space.uppers).all(axis=1))
            costs_to_go = state_space.interpolate(end_states[valid])
            a_idx = np.argmin(costs_to_go)
            action = action_space[valid][a_idx]
            env.step(action, cur_state)

        out.write(np.rot90(env.render()))
        cur_state = env.state

    out.release()

