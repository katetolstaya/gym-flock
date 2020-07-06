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
env = gym.make("FlockingRelative-v0")` 
~~~~
and then use the `env.reset()` and `env.step()` for interfacing with the environment as you would with other OpenAI Gym environments. 
These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.

Please note that the state of these environments returns a tuple for the states of all agents, along with a matrix of the connectivity of the network of agents. 

## Citing the Project
To cite this repository in publications:
```shell
@inproceedings{tolstaya2020learning,
  title={Learning decentralized controllers for robot swarms with graph neural networks},
  author={Tolstaya, Ekaterina and Gama, Fernando and Paulos, James and Pappas, George and Kumar, Vijay and Ribeiro, Alejandro},
  booktitle={Conference on Robot Learning},
  pages={671--682},
  year={2020}
}
```



