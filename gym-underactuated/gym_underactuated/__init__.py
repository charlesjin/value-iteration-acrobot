from gym.envs.registration import register

register(
    id='CustomAcrobot-v0',
    entry_point='gym_underactuated.envs:CustomAcrobotEnv',
)

