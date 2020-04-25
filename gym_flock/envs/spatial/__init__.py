from gym_flock.envs.spatial.coverage import CoverageEnv
from gym_flock.envs.spatial.coverage_arl import CoverageARLEnv
from gym_flock.envs.spatial.coverage_full import CoverageFullEnv

try:
    import airsim
    from gym_flock.envs.spatial.coverage_airsim import CoverageAirsimEnv
except ImportError:
    print('AirSim not installed.')

