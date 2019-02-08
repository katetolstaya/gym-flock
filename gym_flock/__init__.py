from gym.envs.registration import register

register(
    id='flock-v0',
    entry_point='gym_flock.envs:Flock',
)
