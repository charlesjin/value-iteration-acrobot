import gym
import torch
from torch import sin, cos
import torch.nn as nn

from continuous.net import ValueIterNet

acrobot_env = gym.make("gym_underactuated:CustomAcrobot-v0")

class AcrobotNet(ValueIterNet):
    PI = 3.14159265358979323846

    LINK_LENGTH_1 = 1.1  # [m]
    LINK_LENGTH_2 = 2.1  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = .6   #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 1.1  #: [m] position of the center of mass of link 2
    LINK_MOI_1 = 1.  #: moments of inertia for link 1
    LINK_MOI_2 = 1.  #: moments of inertia for link 2

    DAMPING = .1  # damping for both joints

    MAX_VEL_1 = acrobot_env.MAX_VEL_1
    MAX_VEL_2 = acrobot_env.MAX_VEL_2

    TORQUE_LIMIT = acrobot_env.TORQUE_LIMIT

    def __init__(self, step_sizes, **kwargs):
        super().__init__(step_sizes, **kwargs)
        self.env = acrobot_env
        #assert len(step_sizes) == 5

    def get_grid_partials(self, J, dsdt, periodic=[0, 1], order='first', upwind=True):
        return super().get_grid_partials(J, dsdt, periodic, order, upwind)

    @classmethod
    def dsdt(cls, s):
        """
        tensorized version of 
            gym_underactuated.envs.custom_acrobot.CustomAcrobotEnv._dsdt 

        input s is 5 x N1 x N2 x N3 x N4 x N5 tensor
        output is 4 x N1 x N2 x N3 x N4 x N5 tensor
            one derivative for each dimension of the state space
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
        a = s[4]

        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2

        # next line is what the AcrobotEnv calls the ``java'' or ``book'' implementation
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta2 += -d * dtheta2

        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        ddtheta1 += -d * dtheta1

        return torch.stack((dtheta1, dtheta2, ddtheta1, ddtheta2))

    @classmethod
    def cost(cls, s, eps=.1, goal='top', obj='min_time'):
        """
        computes the cost for a given goal and objective

        input s is 5 x N1 x N2 x N3 x N4 x N5 tensor
        input goal can be either `top' or `bottom'
        input obj can be either `min_time' or `lqr'
        output is N1 x N2 x N3 x N4 x N5 tensor
        
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

        r = s[2:-1] # remove position and action
        pos = o*o + s[1]*s[1]
        vel = torch.sum(r*r, dim=0)

        if obj == 'min_time':
            c = (pos > eps) | (vel > eps)
            return c
        elif obj == 'lqr':
            reg = s[-1] * s[-1] * 4
            return pos + vel + reg
        else:
            assert False, "Cost model does not support obj {obj}."

