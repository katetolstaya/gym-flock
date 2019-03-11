from gym.envs.registration import register

register(
    id='Flocking-v0',
    entry_point='gym_flock.envs:FlockingEnv',
    max_episode_steps=200,
)

register(
    id='LQR-v0',
    entry_point='gym_flock.envs:LQREnv',
    max_episode_steps=200,
)
register(
    id='FlockingTest-v0',
    entry_point='gym_flock.envs:FlockingTestEnv',
    max_episode_steps=200,
)

