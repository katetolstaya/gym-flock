import gym
import gym_flock
from gym_flock.envs.spatial.make_map import construct_dist_matrix
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

penalty_multiplier = 1000


def create_data_model():
    data = {}
    # Initialize the gym environment
    env_name = "MappingRad-v0"
    t0 = time.time()
    env = gym.make(env_name)
    t1 = time.time()
    print('Time to construct the environment ' + str(t1 - t0))
    ep_length = env._max_episode_steps
    env = env.env

    # get visitation of nodes
    penalty = np.logical_not(env.visited) * penalty_multiplier
    penalty = np.append(penalty, [0.0])
    data['penalties'] = penalty

    # get map edges from env
    motion_edges = (env.motion_edges[0] - env.n_robots, env.motion_edges[1] - env.n_robots)
    t0 = time.time()
    dist_mat = construct_dist_matrix(motion_edges)
    t1 = time.time()
    print('Time to construct the adjacency matrix ' + str(t1 - t0))
    # add depot at index env.n_targets
    dist_mat = np.vstack((np.zeros((1, env.n_targets)), dist_mat))
    dist_mat = np.hstack((np.zeros((env.n_targets + 1, 1)), dist_mat))
    data['time_matrix'] = dist_mat

    time_window = (0, ep_length)
    data['time_windows'] = [time_window] * (env.n_targets + 1)
    data['num_vehicles'] = env.n_robots
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""

    # Display dropped nodes.
    # dropped_nodes = 'Dropped nodes:'
    n_dropped = 0
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            # dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            n_dropped += 1
    # print(dropped_nodes)
    print('Total number of dropped nodes: ' + str(n_dropped))

    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    assignment.Min(time_var),
                                                    assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        print(plan_output)
        total_time += assignment.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    t0 = time.time()
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time_str = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        30,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time_str)
    time_dimension = routing.GetDimensionOrDie(time_str)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Allow to drop nodes.
    for node in range(1, len(data['time_matrix'])):
        penalty = int(data['penalties'][manager.NodeToIndex(node)])
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    t1 = time.time()
    print('Time to set up the problem ' + str(t1 - t0))

    # Solve the problem.

    t0 = time.time()
    assignment = routing.SolveWithParameters(search_parameters)
    t1 = time.time()
    print('Time to solve the problem ' + str(t1 - t0))

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()
