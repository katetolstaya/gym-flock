import gym
import gym_flock
import numpy as np
import copy
import rospy
from mav_manager.srv import Vec4Request, Vec4
from geometry_msgs.msg import PoseStamped

n_robots = 10
x = np.zeros((n_robots, 2))
names = ['quadrotor' + str(i + 1) for i in range(n_robots)]

altitudes = np.linspace(start=3.0, stop=8.0, num=n_robots)

rospy.init_node('gnn')
# TODO smaller rate here?
r = rospy.Rate(10.0)

env_name = "CoverageARL-v0"

env = gym.make(env_name)
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
env.reset()
env.render()

arl_env = env.env.env

def state_callback(data, robot_index):
    x[robot_index, 0] = data.pose.position.x
    x[robot_index, 1] = data.pose.position.y


for i, name in enumerate(names):
    topic_name = "/unity_ros/" + name + "/TrueState/pose"
    rospy.Subscriber(name=topic_name, data_class=PoseStamped, callback=state_callback, callback_args=i)

services = [rospy.ServiceProxy("/" + name + "/mav_services/goTo", Vec4) for name in names]

while True:

    # update state and get new observation
    arl_env.update_state(x)
    obs, reward, done = arl_env._get_obs_reward()

    # compute local action
    action = arl_env.controller(random=False, greedy=True)
    next_loc = copy.copy(action.reshape((-1, 1)))

    # convert to next waypoint
    for i in range(arl_env.n_robots):
        next_loc[i] = arl_env.mov_edges[1][np.where(arl_env.mov_edges[0] == i)][action[i]]
    loc_commands = np.reshape(arl_env.x[next_loc, 0:2], (arl_env.n_robots, 2))

    # update last loc
    old_last_loc = arl_env.last_loc
    arl_env.last_loc = arl_env.closest_targets

    # send new waypoints
    for i, service in enumerate(services):

        goal_position = [loc_commands[i, 0], loc_commands[i, 1], altitudes[i], -1.57]
        goal_position = Vec4Request(goal_position)
        try:
            service(goal_position)
        except rospy.ServiceException:
            print("Service call failed")

    arl_env.last_loc = np.where(arl_env.last_loc == arl_env.closest_targets, old_last_loc, arl_env.last_loc)

    env.render()
    r.sleep()
