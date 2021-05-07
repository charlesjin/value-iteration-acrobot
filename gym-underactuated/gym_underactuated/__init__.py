from gym.envs.registration import register

register(
    id='CustomAcrobot-v0',
    entry_point='gym_underactuated.envs:CustomAcrobotEnv',
)

register(
    id='CustomCartPole-v0',
    entry_point='gym_underactuated.envs:CustomCartPoleEnv',
)

register(
    id='CustomPendulum-v0',
    entry_point='gym_underactuated.envs:CustomPendulumEnv',
)

