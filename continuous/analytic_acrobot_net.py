import gym
import torch
from torch import sin, cos
import torch.nn as nn
import numpy as np

from continuous.analytic_net import AnalyticValueIterNet
from continuous.acrobot_net import AcrobotNet

class AnalyticAcrobotNet(AcrobotNet, AnalyticValueIterNet):
    def __init__(self, step_sizes, periodic=[0,1], R=0, **kwargs):
        super().__init__(step_sizes, periodic=periodic, R=R, 
                a_max=AcrobotNet.TORQUE_LIMIT, **kwargs)

    def get_grid_partials(self, J, dsdt, periodic=[0, 1], order='first', upwind=True):
        return super().get_grid_partials(J, dsdt, periodic, order, upwind)

    @classmethod
    def dsdt_ddsdtda(cls, s):
        """
        tensorized version of 
            gym_underactuated.envs.custom_acrobot.CustomAcrobotEnv._dsdt 
        assuming that the dynamics are control affine

        ds/dt derivatives are taken at a=0
        then given some control input u, can compute exactly ds/dt at a=u using
            ds/dt|a=a = ds/dt|a=0 + u * d^s/dtda

        input s is 4 x N1 x N2 x N3 x N4 tensor
        output is pair of tensors
            4 x N1 x N2 x N3 x N4 tensor 
                ds/dt|a=0 - one derivative for each dimension of the state space
            4 x N1 x N2 x N3 x N4 tensor 
                d^2s/dtda - one derivative for each dimension of the state space
        """

        m1 = cls.LINK_MASS_1
        m2 = cls.LINK_MASS_2
        l1 = cls.LINK_LENGTH_1
        lc1 = cls.LINK_COM_POS_1
        lc2 = cls.LINK_COM_POS_2
        I1 = cls.LINK_MOI_1
        I2 = cls.LINK_MOI_2
        d = cls.DAMPING
        pi = cls.PI
        g = 9.8

        theta1 = s[0] # pi
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        dtheta1da = torch.zeros(dtheta1.shape).to(dtheta1.device)
        dtheta2da = torch.zeros(dtheta2.shape).to(dtheta1.device)

        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2

        # next line is what the AcrobotEnv calls the ``java'' or ``book'' implementation
        ddtheta2 = (d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta2 += -d * dtheta2
        ddtheta2da = 1 / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)

        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        ddtheta1 += -d * dtheta1
        ddtheta1da = -(d2 * ddtheta2da) / d1

        return torch.stack((dtheta1, dtheta2, ddtheta1, ddtheta2)), \
                torch.stack((dtheta1da, dtheta2da, ddtheta1da, ddtheta2da))

    @classmethod
    def cost_s(cls, s, eps=.1, goal='top', obj='min_time'):
        """
        computes the cost for a given goal and objective
        assumes that the cost can be written as
            l(x, u) = l_1(x) + l_2(u)

        input s is 4 x N1 x N2 x N3 x N4 tensor
        input goal can be either `top' or `bottom'
        input obj either `min_time' or `lqr' (quadratic)
        output N1 x N2 x N3 x N4 tensor for l_1(x)

        e.g., for goal = 'top', obj = 'min_time'
            l(x, u) = 0 if x is upright, 1 otherwise
            the upright position is [pi, 0, 0, 0]
            we will use x.dot(x) > eps = 1e-4
        """

        if goal == 'top':
            o = s[0] - cls.PI
        elif goal == 'bottom':
            o = s[0]
            o[o > cls.PI] -= 2 * cls.PI
        else:
            assert False, "Cost model does not recognize goal {goal}."

        r = s[2:] # remove position
        pos = o*o + s[1]*s[1]
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
        mesh = np.zeros((4, 3, 3, 3, 3))
        mesh[0] += state[0]
        mesh[1] += state[1]
        mesh[2] += state[2]
        mesh[3] += state[3]

        mesh[0,0,:,:,:] -= delta
        mesh[0,2,:,:,:] += delta
        mesh[1,:,0,:,:] -= delta
        mesh[1,:,2,:,:] += delta
        mesh[2,:,:,0,:] -= delta
        mesh[2,:,:,2,:] += delta
        mesh[3,:,:,:,0] -= delta
        mesh[3,:,:,:,2] += delta

        pi = cls.PI
        mesh[0] %= (2*pi)
        mesh[1] = (mesh[1] + pi) % (2*pi) - pi
        mesh[2][mesh[2] >  cls.MAX_VEL_1] =  cls.MAX_VEL_1
        mesh[2][mesh[2] < -cls.MAX_VEL_1] = -cls.MAX_VEL_1
        mesh[3][mesh[3] >  cls.MAX_VEL_2] =  cls.MAX_VEL_2
        mesh[3][mesh[3] < -cls.MAX_VEL_2] = -cls.MAX_VEL_2

        return mesh

