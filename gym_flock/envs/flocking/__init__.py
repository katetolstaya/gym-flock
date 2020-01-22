from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.flocking.flocking import FlockingEnv
from gym_flock.envs.flocking.flocking_leader import FlockingLeaderEnv
from gym_flock.envs.flocking.flocking_obstacle import FlockingObstacleEnv
from gym_flock.envs.flocking.flocking_stoch import FlockingStochasticEnv
from gym_flock.envs.flocking.flocking_twoflocks import FlockingTwoFlocksEnv

try:
    import airsim
    from gym_flock.envs.flocking.flocking_airsim_accel import FlockingAirsimAccelEnv
except ImportError:
    print('AirSim not installed.')
