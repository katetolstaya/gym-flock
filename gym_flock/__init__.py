from gym.envs.registration import register

register(
    id='Flocking-v0',
    entry_point='gym_flock.envs:FlockingEnv',
)

register(
    id='LQR-v0',
    entry_point='gym_flock.envs:LQREnv',
)
