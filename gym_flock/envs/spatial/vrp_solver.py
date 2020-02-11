import gym
import gym_flock
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

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
            # TODO save the index of the minimum neighbor here and then use to reconstruct the path later
            changed_last_iter = changed_last_iter or (not np.array_equal(new_cost, time_matrix[:, receiver]))
            time_matrix[:, receiver] = new_cost

    return time_matrix


def create_data_model(env):
    """
    Formulate the vehicle routing problem corresponding to the MappingRad env to generate the expert solution
    :return: Dict containing the problem parameters
    """
    data = {}
    # data['episode_length'] = env._max_episode_steps
    # env = env.env
    init_loc = env.closest_targets - env.n_robots

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


def solve_vrp(env):
    """

    :param env:
    :type env:
    :return:
    :rtype:
    """
    data = create_data_model(env)
    data['episode_length'] = 50

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
    assignment = routing.SolveWithParameters(search_parameters)

    raw_trajectories = [[]]*data['num_vehicles']
    trajectories = [[]]*data['num_vehicles']

    for vehicle_id in range(data['num_vehicles']):
        index = assignment.Value(routing.NextVar(routing.Start(vehicle_id)))
        # index = assignment.Value(routing.Start(vehicle_id))

        # check conditions on first stops, ignore depot
        assert index in data['init_loc'], 'First stop is not an initial position'
        assert index not in [ls[0] for ls in raw_trajectories if len(ls) > 0]
        result_index = np.where(data['init_loc'] == index)[0].flatten()[0]

        result = []
        raw_result = []

        while not routing.IsEnd(index):
            cur_node = manager.IndexToNode(index)
            proc_node = cur_node - 1 + env.n_robots
            result.append(proc_node)  # remove depot indexing shift
            raw_result.append(cur_node)  # remove depot indexing shift
            index = assignment.Value(routing.NextVar(index))

        trajectories[result_index] = result
        raw_trajectories[result_index] = raw_result
        # don't add depot as last node

    return trajectories


