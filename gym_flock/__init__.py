from gym.envs.registration import register

register(
    id='FlockingRelative-v0',
    entry_point='gym_flock.envs:FlockingRelativeEnv',
    max_episode_steps=200,
)

register(
    id='FlockingLeader-v0',
    entry_point='gym_flock.envs:FlockingLeaderEnv',
    max_episode_steps=200,
)


register(
    id='FlockingObstacle-v0',
    entry_point='gym_flock.envs:FlockingObstacleEnv',
    max_episode_steps=200,
)

register(
    id='FormationFlying-v0',
    entry_point='gym_flock.envs:FormationFlyingEnv',
    max_episode_steps=500,
)

try:
    import airsim
    register(
        id='FlockingAirsim-v0',
        entry_point='gym_flock.envs:FlockingAirsimEnv',
        max_episode_steps=200,
    )
except ImportError:
    print('AirSim not installed.')




# register(
#     id='Flocking-v0',
#     entry_point='gym_flock.envs:FlockingEnv',
#     max_episode_steps=200,
# )

# register(
#     id='FlockingMulti-v0',
#     entry_point='gym_flock.envs:FlockingMultiEnv',
#     max_episode_steps=200,
# )

# register(
#     id='LQR-v0',
#     entry_point='gym_flock.envs:LQREnv',
#     max_episode_steps=200,
# )
# register(
#     id='FlockingTest-v0',
#     entry_point='gym_flock.envs:FlockingTestEnv',
#     max_episode_steps=200,
# )

# register(
#     id='Consensus-v0',
#     entry_point='gym_flock.envs:ConsensusEnv',
#     max_episode_steps=500,
# )


