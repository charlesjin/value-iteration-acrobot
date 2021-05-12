import cv2
import numpy as np
from numpy import pi
import multiprocessing as mp
import imageio
import os
from gym_underactuated.envs import *

seconds = 20

def add_text(env, view, state, action, ctg=None, cost=None, warn_text=None):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = .4 
    fontColor              = 255
    thickness              = 1

    if isinstance(env, CustomAcrobotEnv):
        text = [f"theta1:    {state[0]: .3f}",
                f"theta2:    {state[1]: .3f}",
                f"dtheta1:   {state[2]: .3f}",
                f"dtheta2:   {state[3]: .3f}",
                f"tau:       {action: .3f}"]
    elif isinstance(env, CustomPendulumEnv):
        text = [f"theta:     {state[0]: .3f}",
                f"dtheta:    {state[1]: .3f}",
                f"tau:       {action: .3f}"]
    else:
        text = []
    if ctg is not None:
        text.append(
            f"ctg:       {ctg: .3f}")

    if cost is not None:
        text.append(
            f"cost:      {cost: .3f}")

    start = 490-15*len(text)
    for i, t in enumerate(text):
        bottomLeftCornerOfText = (20,start+15*i)
        view = cv2.putText(view, t, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness=thickness)

    if warn_text is not None:
        bottomLeftCornerOfText = (20,25)
        view = cv2.putText(view, warn_text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness=thickness)

    return view

def add_zero_text(env, view, state, time):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = .4 
    fontColor              = 255
    thickness              = 1

    text = [f"cost=0 at time={time:.3f}"]
    if isinstance(env, CustomAcrobotEnv):
        text.extend([f"theta1:    {state[0]: .3f}",
                f"theta2:    {state[1]: .3f}",
                f"dtheta1:   {state[2]: .3f}",
                f"dtheta2:   {state[3]: .3f}"])
    elif isinstance(env, CustomPendulumEnv):
        text.extend([f"theta:     {state[0]: .3f}",
                f"dtheta:    {state[1]: .3f}"])

    for i, t in enumerate(text):
        bottomLeftCornerOfText = (20,25+15*i)
        view = cv2.putText(view, t, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness=thickness)

    return view

def _sim(fn, start_state, env, state_space, action_space, use_policy, cost_fn, policy_fn):
    if state_space is not None:
        assert action_space is not None
    if use_policy:
        assert state_space.data_dims > 1

    fps = int (1 / env.dt)
    skip_frames = fps // 30 + 1

    try:
        #out = cv2.VideoWriter(f'{fn}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (500, 500), False)
        out = imageio.get_writer(f'{fn}.tmp.gif', mode='I', duration=env.dt * skip_frames)

        env.state = start_state

        ctg = state_space.interpolate(np.array([start_state]))[0]
        start_frame = cv2.rotate(env.render(), cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cost = cost_fn(start_state, 0)
        warn_text = f"cost=0 at time=X"
        start_frame = add_text(env, start_frame, start_state, 0, ctg, cost, warn_text)
        for i in range(fps // 2):
            if i % skip_frames == 0:
                #out.write(start_frame)
                out.append_data(start_frame)

        mu = 0 
        zero_state = None
        for i in range(fps * seconds):
            cur_state = env.state
            if state_space is None:
                # simulate the natural dynamics
                action = 0 
                ctg = None
                env.step(action, cur_state)
            elif use_policy:
                weights, a_idxs = \
                        state_space.interpolate(np.array([cur_state]), 
                                                data_dim=1, dot=False)

                weights, a_idxs = weights[0], a_idxs[0]
                a_idx = np.sum(a_idxs * weights)
                ctg = state_space.interpolate(np.array([cur_state]))[0]

                #####
                # FOR NON ANALYTIC

                #if max(a_idxs[weights > .01]) - min(a_idxs[weights > .01]) >= 10:
                #    #print(max(a_idxs))
                #    #print(min(a_idxs))
                #    #a_idx = np.sum(a_idxs * weights)
                #    #print(a_idx)
                #    if a_idx > 10:
                #        weights[a_idxs < 10] = 0
                #    else:
                #        weights[a_idxs > 10] = 0
                #    weights /= np.sum(weights)
                #    a_idx = np.sum(a_idxs * weights)
                #    #print(a_idx)

                #ctg = state_space.interpolate(np.array([cur_state]))[0]
                #low = int(a_idx)
                #d = a_idx - low
                #high = low + 1

                #high = min(high, len(action_space)-1)
                #low = min(low, len(action_space)-1)
                #action = action_space[low] * (1 - d) + action_space[high] * d
                # END
                #####

                ######
                ## FOR ANALYTIC
                action = a_idx
                ## END
                ######

                env.step(action, cur_state)
            elif policy_fn is not None:
                delta = .01
                u0 = d0 = u1 = d1 = u2 = d2 = u3 = d3 = cur_state
                u0[0] += delta
                d0[0] -= delta
                u1[1] += delta
                d1[1] -= delta
                u2[2] += delta
                u2[2] -= delta
                u3[3] += delta
                u3[3] -= delta
                states = [cur_state, u0, d0, u1, d1, u2, d2, u3, d3]

                weights, Js = \
                        state_space.interpolate(np.array([states]), 
                                                data_dim=0, dot=False)
                J, Js = Js[0], Js[1:]

                local_J = torch.zeros([3,3,3,3])
                local_J[1,1,1,1] = J

                Js = (Js - J) / delta
                Js[0:2] *= state_space.step_sizes[0]
                Js[2:4] *= state_space.step_sizes[1]
                Js[4:6] *= state_space.step_sizes[2]
                Js[6:8] *= state_space.step_sizes[3]
                Js += J
                
                local_J[2,1,1,1] = Js[0]
                local_J[0,1,1,1] = Js[1]
                local_J[1,2,1,1] = Js[2]
                local_J[1,0,1,1] = Js[3]
                local_J[1,1,2,1] = Js[4]
                local_J[1,1,0,1] = Js[5]
                local_J[1,1,1,2] = Js[6]
                local_J[1,1,1,0] = Js[7]

                action = policy_fn(local_J, cur_state)
                ctg = J
                env.state = env.step(action, cur_state)

            else:
                assert action_space is not None

                # simulate the ctg controller
                end_states = \
                        np.array([env.step(action, cur_state)[0] for action in action_space])
                valid = np.logical_and((end_states >= state_space.lowers).all(axis=1), 
                                       (end_states <= state_space.uppers).all(axis=1))

                costs_to_go = state_space.interpolate(end_states[valid])

                #valid_actions = action_space[valid]
                #u_cost = valid_actions.dot(valid_actions) / 1000
                #a_idx = np.argmin(costs_to_go + u_cost)
                a_idx = np.argmin(costs_to_go)

                action = action_space[valid][a_idx]
                ctg = costs_to_go[a_idx]
                env.state = end_states[valid][a_idx]

                #idxs = state_space.interpolate(end_states[valid])
                #costs_to_go = state_space.data[0][idxs]
                #a_idx = np.argmin(costs_to_go)
                #coord = [x[a_idx] for x in idxs]
                #action = action_space[valid][a_idx]
                #ctg = costs_to_go[a_idx]

                #flat_coord = np.ravel_multi_index(coord, state_space.steps+1)
                #cur_state = state_space.points[flat_coord,:]
                #env.state = cur_state

            cost = cost_fn(cur_state, action)
            if cost == 0:
                zero_state = cur_state
                zero_time = i * env.dt
                warn_text = None

            if i % skip_frames == 0:
                view = cv2.rotate(env.render(action), cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                view = add_text(env, view, cur_state, action, ctg, cost, warn_text)
                if zero_state is not None:
                    view = add_zero_text(env, view, zero_state, zero_time)

                #out.write(view)
                out.append_data(view)


        #out.release()
        out.close()
        os.rename(f'{fn}.tmp.gif', f'{fn}.gif')
    except KeyboardInterrupt:
        os.remove(f'{fn}.tmp.gif')

def sim(fn, start_state, env, 
        state_space=None, action_space=None, 
        use_policy=False, cost_fn=None, 
        policy_fn=None,
        start_proc=True):
    if start_proc:
        p = mp.Process(target=_sim, args=(fn, start_state, env, state_space, action_space, use_policy, cost_fn, policy_fn))
        p.start()
        return p
    else:
        _sim(fn, start_state, env, state_space, action_space, use_policy, cost_fn, policy_fn)

