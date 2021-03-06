import gym
import torch
from torch import sin, cos
import numpy as np

from continuous.net import ValueIterNet

pendulum_env = gym.make("gym_underactuated:CustomPendulum-v0")

class PendulumNet(ValueIterNet):
    PI = 3.14159265358979323846

    LINK_LENGTH = 1.  # [m]
    LINK_MASS = 1.  #: [kg] mass of link 1
    DAMPING = .1  # damping for both joints

    MAX_VEL = pendulum_env.max_speed
    TORQUE_LIMIT = pendulum_env.max_torque
    G = pendulum_env.g

    def __init__(self, step_sizes, **kwargs):
        super().__init__(step_sizes, **kwargs)
        self.env = pendulum_env
        #assert len(step_sizes) == 3

    def get_grid_partials(self, J, dsdt, periodic=[0], order='first', upwind=True):
        return super().get_grid_partials(J, dsdt, periodic, order, upwind)

    @classmethod
    def dsdt(cls, s):
        """
        input s is 3 x N1 x N2 x N3 tensor
        output is 2 x N1 x N2 tensor
            one derivative for each dimension of the state space
        """

        l = cls.LINK_LENGTH
        m = cls.LINK_MASS
        d = cls.DAMPING
        pi = cls.PI
        g = 9.8

        theta = s[0]
        dtheta = s[1]
        a = s[2]

        ddtheta = \
                - 3. * g / (2 * l) * sin(theta + pi) \
                + 3. / (m * l ** 2) * a \
                - d * dtheta

        return torch.stack((dtheta, ddtheta))

    @classmethod
    def cost(cls, s, eps=.0001, goal='top', obj='min_time'):
        """
        computes the cost for a given goal and objective

        input s is 3 x N1 x N2 x N3 tensor
        input goal can be either `top' or `bottom'
        input obj can be either `min_time' or `lqr'
        output is N1 x N2 x N3 tensor
        
        e.g., for goal = 'top', obj = 'min_time'
            l(x, u) = 0 if x is upright, 1 otherwise
            the upright position is [0, 0]
            we will use x.dot(x) > eps
        """

        # theta : [-pi, pi], with 0 being the top
        if goal == 'top':
            o = s[0]
        elif goal == 'bottom':
            o = s[0] + cls.PI
            o[o > cls.PI] -= 2 * cls.PI
        else:
            assert False, "Cost model does not recognize goal {goal}."

        r = s[1:-1] # remove action
        pos = (o*o + torch.sum(r*r, dim=0))

        if obj == 'min_time':
            return pos > eps
        elif obj == 'lqr':
            reg = s[-1] * s[-1] * 4
            #reg /= torch.max(reg)
            return pos + reg
        else:
            assert False, "Cost model does not support obj {obj}."

    @classmethod
    def get_state_mesh(cls, state, delta=.01): 
        mesh = np.zeros((3, 3, 3, 3))
        mesh[0] += state[0]
        mesh[1] += state[1]

        mesh[2,:,:,0] = -cls.TORQUE_LIMIT
        mesh[2,:,:,1] = 0
        mesh[2,:,:,2] = cls.TORQUE_LIMIT 

        mesh[0,0,:] -= delta
        mesh[0,2,:] += delta
        mesh[1,:,0] -= delta
        mesh[1,:,2] += delta

        pi = cls.PI
        mesh[0] = (mesh[0] + pi) % 2*pi - pi
        mesh[1][mesh[1] >  cls.MAX_VEL] =  cls.MAX_VEL
        mesh[1][mesh[1] < -cls.MAX_VEL] = -cls.MAX_VEL

        return mesh

    def action_single(self, state, interpolate_fn, eps, gamma, delta=.01):
        states_and_actions = self.get_state_mesh(state, delta)
        states = states_and_actions[0:2,:,:,0]

        J = interpolate_fn(np.transpose(states.reshape(states.shape[0], -1)))
        J = torch.Tensor(J.reshape(states.shape[1:]))

        states = torch.Tensor(states_and_actions)
        dsdt = self.dsdt(states)
        cost = self.cost(states, eps=eps)

        step_sizes = self.step_sizes
        self.step_sizes = torch.Tensor([delta] * len(J.shape))
        dJdt, a = self.forward(J, dsdt, cost, gamma)
        self.step_sizes = step_sizes

        while len(dJdt.shape) > 1:
            dJdt = dJdt[1]
            a = a[1]
        return dJdt[1].item(), a[1].item()

