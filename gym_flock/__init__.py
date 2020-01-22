from gym.envs.registration import register

register(
    id='Shepherding-v0',
    entry_point='gym_flock.envs.shepherding:ShepherdingEnv',
    max_episode_steps=1000,
)

register(
    id='MappingRad-v0',
    entry_point='gym_flock.envs.spatial:MappingRadEnv',
    max_episode_steps=100,
)

register(
    id='Flocking-v0',
    entry_point='gym_flock.envs.flocking:FlockingEnv',
    max_episode_steps=1000,
)

register(
    id='FlockingRelative-v0',
    entry_point='gym_flock.envs.flocking:FlockingRelativeEnv',
    max_episode_steps=1000,
)

register(
    id='FlockingLeader-v0',
    entry_point='gym_flock.envs.flocking:FlockingLeaderEnv',
    max_episode_steps=200,
)


register(
    id='FlockingObstacle-v0',
    entry_point='gym_flock.envs.flocking:FlockingObstacleEnv',
    max_episode_steps=200,
)

register(
    id='FormationFlying-v0',
    entry_point='gym_flock.envs.formation:FormationFlyingEnv',
    max_episode_steps=500,
)

register(
    id='FlockingStochastic-v0',
    entry_point='gym_flock.envs.flocking:FlockingStochasticEnv',
    max_episode_steps=500,
)

register(
    id='FlockingTwoFlocks-v0',
    entry_point='gym_flock.envs.flocking:FlockingTwoFlocksEnv',
    max_episode_steps=500,
)


try:
    import airsim

    register(
        id='FlockingAirsimAccel-v0',
        entry_point='gym_flock.envs.flocking:FlockingAirsimAccelEnv',
        max_episode_steps=200,
    )
except ImportError:
    print('AirSim not installed.')


