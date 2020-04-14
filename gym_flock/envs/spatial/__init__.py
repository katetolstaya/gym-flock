from gym_flock.envs.spatial.mapping_rad import MappingRadEnv
from gym_flock.envs.spatial.mapping_arl_partial import MappingARLPartialEnv

try:
    import airsim
    from gym_flock.envs.spatial.mapping_airsim import MappingAirsimEnv
except ImportError:
    print('AirSim not installed.')

