from gym_flock.envs.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.flocking import FlockingEnv
from gym_flock.envs.mapping import MappingEnv
from gym_flock.envs.mapping_local import MappingLocalEnv
from gym_flock.envs.flocking_obstacle import FlockingObstacleEnv
from gym_flock.envs.flocking_leader import FlockingLeaderEnv
from gym_flock.envs.formation_flying import FormationFlyingEnv
from gym_flock.envs.flocking_stoch import FlockingStochasticEnv
from gym_flock.envs.flocking_twoflocks import FlockingTwoFlocksEnv

try:
	import airsim
	# from gym_flock.envs.old.flocking_airsim import FlockingAirsimEnv
	from gym_flock.envs.flocking_airsim_accel import FlockingAirsimAccelEnv
except ImportError:
	print('AirSim not installed.')


