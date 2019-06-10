
from gym_flock.envs.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.flocking_obstacle import FlockingObstacleEnv
from gym_flock.envs.flocking_leader import FlockingLeaderEnv
from gym_flock.envs.formation_flying import FormationFlyingEnv

# from gym_flock.envs.flocking import FlockingEnv
# from gym_flock.envs.flocking_multi import FlockingMultiEnv
# from gym_flock.envs.lqr import LQREnv
# from gym_flock.envs.flocking_test import FlockingTestEnv
# from gym_flock.envs.consensus import ConsensusEnv


try:
	import airsim
	from gym_flock.envs.flocking_airsim import FlockingAirsimEnv
except ImportError:
	print('AirSim not installed.')


