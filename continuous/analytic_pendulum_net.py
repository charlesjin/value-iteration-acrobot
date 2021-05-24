import gym
import torch
from torch import sin, cos
import torch.nn as nn
import numpy as np

from continuous.analytic_net import AnalyticValueIterNet
from continuous.pendulum_net import PendulumNet

class AnalyticPendulumNet(AnalyticValueIterNet, PendulumNet):
    def __init__(self, step_sizes, periodic=[0], R=0, **kwargs):
        super().__init__(step_sizes, periodic=periodic, R=R, 
                a_max=PendulumNet.TORQUE_LIMIT, **kwargs)

    def get_grid_partials(self, J, dsdt, periodic=[0], order='first', upwind=True):
        return super().get_grid_partials(J, dsdt, periodic, order, upwind)

    @classmethod
    def dsdt_ddsdtda(cls, s):
        """
        input s is 2 x N1 x N2 tensor
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

        ddtheta = \
                - 3. * g / (2 * l) * sin(theta + pi) \
                - d * dtheta

        dthetada = torch.zeros(dtheta.shape).to(dtheta.device)
        ddthetada = torch.zeros(dtheta.shape).to(dtheta.device)
        ddthetada += 3. / (m * l ** 2)

        return torch.stack((dtheta, ddtheta)), \
                torch.stack((dthetada, ddthetada))

    @classmethod
    def cost_s(cls, s, eps=.1, goal='top', obj='min_time'):
        """
        computes the cost for a given goal and objective
        assumes that the cost can be written as
            l(x, u) = l_1(x) + l_2(u)
        """

        # theta : [-pi, pi], with 0 being the top
        if goal == 'top':
            o = s[0]
        elif goal == 'bottom':
            o = s[0] + cls.PI
            o[o > cls.PI] -= 2 * cls.PI
        else:
            assert False, "Cost model does not recognize goal {goal}."

        r = s[1:] # remove position
        pos = o*o 
        vel = torch.sum(r*r, dim=0)

        if obj == 'min_time':
            c = ((pos > eps) | (vel > eps)).to(torch.uint8)
            return c
        elif obj == 'lqr':
            return pos + vel
        else:
            assert False, "Cost model does not support obj {obj}."

    @classmethod
    def get_state_mesh(cls, state, delta=.01): 
        mesh = np.zeros((2, 3, 3))
        mesh[0] += state[0]
        mesh[1] += state[1]

        mesh[0,0,:] -= delta
        mesh[0,2,:] += delta
        mesh[1,:,0] -= delta
        mesh[1,:,2] += delta

        pi = cls.PI
        mesh[0] = (mesh[1] + pi) % (2*pi) - pi
        mesh[1][mesh[1] >  cls.MAX_VEL] =  cls.MAX_VEL
        mesh[1][mesh[1] < -cls.MAX_VEL] = -cls.MAX_VEL

        return mesh

