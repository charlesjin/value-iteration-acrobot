"""classic Acrobot task"""
import numpy as np
from scipy import integrate
from numpy import sin, cos, pi

from gym import core, spaces
from gym.envs import classic_control
from gym.utils import seeding

import cv2

# SOURCE:
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class CustomAcrobotEnv(classic_control.AcrobotEnv):

    """
    Custom port of the OpenAI gym environment for Acrobot
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    # trying to match
    # https://github.com/RobotLocomotion/drake/blob/master/examples/acrobot/Acrobot.urdf
    LINK_LENGTH_1 = 1.1  # [m]
    LINK_LENGTH_2 = 2.1  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = .6   #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 1.1  #: [m] position of the center of mass of link 2
    LINK_MOI_1 = 1.  #: moments of inertia for link 1
    LINK_MOI_2 = 1.  #: moments of inertia for link 2

    DAMPING = .1  # damping for both joints

    MAX_VEL_1 = 9 * pi
    MAX_VEL_2 = 9 * pi

    #AVAIL_TORQUE = [-1., 0., +1]
    TORQUE_LIMIT = 10.

    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        high = np.array([self.TORQUE_LIMIT], dtype=np.float32)
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        #self.action_space = spaces.Discrete(3)

        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def step(self, a, s=None):
        s = self.state if s is None else s
        torque = bound(a, -self.TORQUE_LIMIT, self.TORQUE_LIMIT)

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)
        s_augmented[0] = wrap(s_augmented[0], 0, 2*pi)
        s_augmented[1] = wrap(s_augmented[1], -pi, pi)

        integrator = integrate.RK45(self._dsdt, 0, s_augmented, self.dt)
        while not integrator.status == 'finished':
            integrator.step()
        ns = integrator.y

        #ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        #ns = ns[-1]

        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        #ns[0] = wrap(ns[0], -pi, pi)
        #ns[1] = wrap(ns[1], -pi, pi)
        ns[0] = wrap(ns[0], 0, 2*pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        #return (self._get_ob(), reward, terminal, {})

        return (self.state, reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        theta1 = s[0] - pi
        theta2 = s[1] #- pi
        err = s.dot(s)
        return bool(err < .01 ** 2)

    #def _dsdt(self, s_augmented, t):
    def _dsdt(self, t, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        d = self.DAMPING
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        # add damping
        ddtheta1 += -d * dtheta1
        ddtheta2 += -d * dtheta2

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode=None):
        s = self.state
        if s is None: return None

        view = np.zeros((500, 500), dtype=np.uint8)
        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = 500 / bound / 2 # scaling from real units to view units

        circle_rad = .1
        line_thick = .05

        p0 = [0,0]
        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), 
              self.LINK_LENGTH_1 * sin(s[0])]
        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        # draw circles
        for p in [p0, p1, p2]:
            view = cv2.circle(view, tuple([int(_p * scale) + 250 for _p in p0]),
                              int(circle_rad * scale), 255, -1)
        # draw lines
        for l in [(p0, p1), (p1, p2)]:
            view = cv2.line(view, 
                            tuple([int(_l * scale) + 250 for _l in l[0]]),
                            tuple([int(_l * scale) + 250 for _l in l[1]]),
                            128, int(line_thick * scale))

        return view

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

