import airsim
import numpy as np
from time import sleep
import re


def get_states(client, names, home):
    n_agents = len(names)
    states = np.zeros(shape=(n_agents, 4))
    yaws = np.zeros(shape=(n_agents, 1))
    for i in range(0, n_agents):
        state = client.getMultirotorState(vehicle_name=names[i])
        states[i][0] = float(state.kinematics_estimated.position.x_val) + home[i][0]
        states[i][1] = float(state.kinematics_estimated.position.y_val) + home[i][1]
        states[i][2] = float(state.kinematics_estimated.linear_velocity.x_val)
        states[i][3] = float(state.kinematics_estimated.linear_velocity.y_val)

        yaws[i] = quaternion_to_yaw(state.kinematics_estimated.orientation)

    return states, yaws


def setup_drones(client, names):
    n_agents = len(names)
    for i in range(0, n_agents):
        client.enableApiControl(True, names[i])
    for i in range(0, n_agents):
        client.armDisarm(True, names[i])

    fi = []
    for i in range(n_agents):
        fi.append(client.takeoffAsync(vehicle_name=names[i]))  # .join()
    for f in fi:
        f.join()


def send_accel_commands(client, names, z, u, duration=0.01):
    n_agents = len(names)
    fi = []
    for i in range(n_agents):
        fi.append(client.moveByAngleZAsync(float(u[i, 0]), float(u[i, 1]), z[i], 0.0, duration,
                                           vehicle_name=names[i]))
    for f in fi:
        f.join()


def send_velocity_commands(client, names, z, u, duration=0.01):
    n_agents = len(names)
    fi = []
    for i in range(n_agents):
        fi.append(client.moveByVelocityZAsync(float(u[i, 0]), float(u[i, 1]), z[i], duration,
                                              vehicle_name=names[i]))
    for f in fi:
        f.join()


def send_loc_commands(client, names, home, loc, z, timeout=5):
    n_agents = len(names)
    fi = []
    for i in range(n_agents):
        fi.append(client.moveToPositionAsync(loc[i][0] - home[i][0],
                                             loc[i][1] - home[i][1], z[i], 6.0,
                                             vehicle_name=names[i]))
    sleep(0.1)
    for f in fi:
        f._timeout = timeout  # quads sometimes get stuck during a crash and never reach the destination
        f.join()


def display_msg(client, msg):
    print(msg)
    client.simPrintLogMessage(msg)


def quaternion_to_yaw(q):
    # yaw (z-axis rotation) from quaternion
    w = float(q.w_val)
    x = float(q.x_val)
    y = float(q.y_val)
    z = float(q.z_val)
    siny_cosp = +2.0 * (w * z + x * y)
    cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def parse_settings(fname):
    names = []
    homes = []
    for line in open(fname):
        for n in re.findall(r'\"(.+?)\": {', line):
            if n != 'Vehicles':
                names.append(n)
        p = re.findall(r'"X": ([-+]?\d*\.*\d+), "Y": ([-+]?\d*\.*\d+), "Z": ([-+]?\d*\.*\d+)', line)
        if p:
            homes.append(np.array([float(p[0][0]), float(p[0][1]), float(p[0][2])]).reshape((1, 3)))
    return names, np.concatenate(homes, axis=0)
