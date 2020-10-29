from gym_flock.envs.spatial.coverage import CoverageEnv
from gym_flock.envs.spatial.coverage_arl import CoverageARLEnv
from gym_flock.envs.spatial.coverage_full import CoverageFullEnv
from gym_flock.envs.spatial.coverage_explore import ExploreEnv
from gym_flock.envs.spatial.coverage_explore_full import ExploreFullEnv


try:
    import airsim
    from gym_flock.envs.spatial.coverage_airsim import CoverageAirsimEnv
except ImportError:
    print('AirSim not installed.')

