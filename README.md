# Gym Flock

## Dependencies
- [OpenAI Gym](https://github.com/openai/gym) 0.11.0
- Python 3 (Python 2 doesn't work)
- [AirSim](https://github.com/microsoft/AirSim) (optional)

## To install
1) Clone this repository
2) `cd gym-flock`
2) `pip3 install -e . `

## Quick test
In the `gym-flock` folder:
1) `python3 test.py`

## To use

Include the following code in your Python script:
~~~~
import gym  
import gym_flock 
env = gym.make("Coverage-v0")` 
~~~~
and then use the `env.reset()` and `env.step()` for interfacing with the environment as you would with other OpenAI Gym environments. 
These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.

The learning code for the flocking task can be found [here](https://github.com/katetolstaya/multiagent_gnn_policies)

The learning code for the spatial coverage tasks can be found [here](https://github.com/katetolstaya/graph_rl)

## Using AirSim

Install AirSim according to instructions [here](https://github.com/microsoft/AirSim)

Move the settings.json file from `gym-flock/gym_flock/envs/airsim/` to `~/Documents/AirSim/`
Try changing the configuration to change the number of robots. The default number of robots is 5.

Launch AirSim

Launch test code with environment name `FlockingAirsimAccel-v0`. The code for this environment can be found [here](https://github.com/katetolstaya/gym-flock/blob/stable/gym_flock/envs/flocking/flocking_airsim_accel.py)
The environment will read the configuration file to identify the number of robots in the simulation.

## Citing the Project
To cite the flocking environment in publications:
```shell
@inproceedings{tolstaya2020learning,
  title={Learning decentralized controllers for robot swarms with graph neural networks},
  author={Tolstaya, Ekaterina and Gama, Fernando and Paulos, James and Pappas, George and Kumar, Vijay and Ribeiro, Alejandro},
  booktitle={Conference on Robot Learning},
  pages={671--682},
  year={2020}
}
```
To cite the spatial coverage code in publications:
```shell
@misc{tolstaya2020multirobot,
      title={Multi-Robot Coverage and Exploration using Spatial Graph Neural Networks}, 
      author={Ekaterina Tolstaya and James Paulos and Vijay Kumar and Alejandro Ribeiro},
      year={2020},
      eprint={2011.01119},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```



