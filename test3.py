import gym
import gym_flock
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

penalty_multiplier = 1000


def construct_time_matrix(edges, edge_time=1.0):
    """
    Compute the distance between all pairs of nodes in the graph
    :param edges: list of edges provided as (sender, receiver)
    :param edge_time: uniform edge cost, assumed to be 1.0
    :return:
    """
    n_nodes = int(max(max(edges[0]), max(edges[1])) + 1)
    time_matrix = np.ones((n_nodes, n_nodes)) * np.Inf
    np.fill_diagonal(time_matrix, 0.0)

    changed_last_iter = True  # prevents looping forever in disconnected graphs

    while changed_last_iter and np.sum(time_matrix) == np.Inf:
        changed_last_iter = False
        for i, (sender, receiver) in enumerate(zip(edges[0], edges[1])):
            new_cost = np.minimum(time_matrix[:, sender] + edge_time, time_matrix[:, receiver])
            changed_last_iter = changed_last_iter or (not np.array_equal(new_cost, time_matrix[:, receiver]))
            time_matrix[:, receiver] = new_cost

    return time_matrix


def create_data_model():
    """
    Formulate the vehicle routing problem corresponding to the MappingRad env to generate the expert solution
    :return: Dict containing the problem parameters
    """
    data = {}

    # Initialize the gym environment
    env_name = "MappingRad-v0"
    env = gym.make(env_name)
    data['episode_length'] = env._max_episode_steps
    env = env.env
    init_loc = env.nearest_landmarks

    # get visitation of nodes
    penalty = np.logical_not(env.visited) * penalty_multiplier
    penalty = np.append(penalty, [0.0])
    data['penalties'] = penalty

    # get map edges from env
    motion_edges = (env.motion_edges[0] - env.n_robots, env.motion_edges[1] - env.n_robots)
    dist_mat = construct_time_matrix(motion_edges)

    # add depot at index env.n_targets with distance = 0 to/from all nodes
    from_depot = np.ones((1, env.n_targets)) * 1000.0
    from_depot[:, init_loc] = 0.0

    to_depot = np.zeros((env.n_targets + 1, 1))

    dist_mat = np.vstack((from_depot, dist_mat))
    dist_mat = np.hstack((to_depot, dist_mat))
    data['time_matrix'] = dist_mat

    data['num_vehicles'] = env.n_robots
    data['init_loc'] = init_loc + 1
    data['depot'] = 0

    print('Number of robots: ' + str(env.n_robots))
    print('Number of targets: ' + str(env.n_targets))
    print('Initial locations: ' + str(data['init_loc']))
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""

    # Display dropped nodes.
    dropped_nodes = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes.append(manager.IndexToNode(node))
    penalty = int(np.sum(data['penalties'][dropped_nodes]) / penalty_multiplier)
    num_unvisited = int(np.sum(data['penalties']) / penalty_multiplier)
    print('Number of nodes dropped: ' + str(len(dropped_nodes)))
    print('Number of unvisited nodes dropped: ' + str(penalty) + ' out of ' + str(num_unvisited))

    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    first_locs = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)

        # check conditions on first stops
        first_loc = assignment.Value(routing.NextVar(index))
        assert first_loc in data['init_loc'], 'First stop is not an initial position'
        assert first_loc not in first_locs, 'First stop is not unique'
        first_locs.append(first_loc)

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
        data['episode_length'],  # allow waiting time
        data['episode_length'],  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time_str)
    time_dimension = routing.GetDimensionOrDie(time_str)

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

    # Anytime search parameters:
    # search_parameters.time_limit.seconds = 30
    # search_parameters.solution_limit = 100

    # Solve the problem.

    t0 = time.time()
    assignment = routing.SolveWithParameters(search_parameters)
    t1 = time.time()

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)

    print('Time to solve the problem ' + str(t1 - t0))


if __name__ == '__main__':
    main()
