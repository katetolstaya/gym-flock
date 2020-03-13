import numpy as np
import matplotlib.pyplot as plt
# from gym_flock.envs.spatial.utils import _get_graph_edges, _get_pos_diff
from scipy.spatial import Delaunay

def in_obstacle(obstacles, px, py):
    """
    Check if query point is within any of the rectangular obstacles
    :param obstacles: list of rectangular obstacles [(xmin, xmax, ymin, ymax)]
    :param px: query point x coordinate
    :param py: query point y coordinate
    :return:
    """
    for (xmin, xmax, ymin, ymax) in obstacles:
        if xmin <= px <= xmax and ymin <= py <= ymax:
            return True
    return False


def gen_obstacle_grid(ranges):
    obstacle_list = []
    for (x1, x2) in ranges:
        for (y1, y2) in ranges:
            obstacle_list.append((x1, x2, y1, y2))
    return obstacle_list


def generate_lattice(free_region, lattice_vectors):
    """
    Generate hexagonal lattice
    From https://stackoverflow.com/questions/6141955/efficiently-generate-a-lattice-of-points-in-python
    :param free_region:
    :param lattice_vectors:
    :return:
    """
    (xmin, xmax, ymin, ymax) = free_region
    image_shape = np.array([xmax - xmin, ymax - ymin])
    center_pix = image_shape // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = image_shape[0] // dx_cell
    ny = image_shape[1] // dy_cell
    # Generate a square lattice, with too many points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    x_lattice = lattice_vectors[0][0] * x_sq + lattice_vectors[1][0] * y_sq
    y_lattice = lattice_vectors[0][1] * x_sq + lattice_vectors[1][1] * y_sq
    # Trim to fit in box.
    mask = ((x_lattice < image_shape[0] / 2.0) & (x_lattice > -image_shape[0] / 2.0))
    mask = mask & ((y_lattice < image_shape[1] / 2.0) & (y_lattice > -image_shape[1] / 2.0))
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # Translate to the center pix.
    x_lattice += (center_pix[0] + xmin)
    y_lattice += (center_pix[1] + ymin)
    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = y_lattice
    out[:, 1] = x_lattice
    return out


def reject_collisions(points, obstacles=None):
    """

    :param points:
    :param obstacles:
    :return:
    """
    if obstacles is None or len(obstacles) is 0:
        return points

    # remove points within obstacle
    n_points = np.shape(points)[0]
    flag = np.ones((n_points,), dtype=np.bool)
    for i in range(n_points):
        if in_obstacle(obstacles, points[i, 0], points[i, 1]):
            flag[i] = False

    return points[flag, :]


def gen_square(env):
    env.x_max = env.x_max_init * env.n_agents / 4
    env.y_max = env.y_max_init * env.n_agents / 4
    per_side = int(env.n_targets / 4)

    targets = set()

    # initialize fixed grid of targets
    tempx = np.linspace(-env.x_max, -env.x_max, 1)
    tempy = np.linspace(-env.y_max, env.y_max, per_side, endpoint=False)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(env.x_max, env.x_max, 1)
    tempy = np.linspace(-env.y_max, env.y_max, per_side, endpoint=False)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(-env.x_max, env.x_max, per_side, endpoint=False)
    tempy = np.linspace(env.y_max, env.y_max, 1)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(-env.x_max, env.x_max, per_side, endpoint=False)
    tempy = np.linspace(-env.y_max, -env.y_max, 1)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))
    targets.add((env.x_max, env.y_max))

    targets = list(zip(*targets))

    env.x[env.n_robots:, 0] = targets[0]
    env.x[env.n_robots:, 1] = targets[1]


def gen_grid(env):
    env.n_targets_side = int(np.sqrt(env.n_targets))
    env.x_max = env.x_max_init * env.n_targets_side
    env.y_max = env.y_max_init * env.n_targets_side
    tempx = np.linspace(-1.0 * env.x_max, env.x_max, env.n_targets_side)
    tempy = np.linspace(-1.0 * env.y_max, env.y_max, env.n_targets_side)
    tx, ty = np.meshgrid(tempx, tempy)
    env.x[env.n_robots:, 0] = tx.flatten()
    env.x[env.n_robots:, 1] = ty.flatten()


def gen_sparse_grid(env):
    env.x_max = env.x_max_init * env.n_agents / 6
    env.y_max = env.y_max_init * env.n_agents / 6

    per_side = int(env.n_targets / 6)

    targets = set()

    # initialize fixed grid of targets
    tempx = np.linspace(-env.x_max, -env.x_max, 1)
    tempy = np.linspace(-env.y_max, env.y_max, per_side, endpoint=False)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(env.x_max, env.x_max, 1)
    tempy = np.linspace(-env.y_max, env.y_max, per_side, endpoint=False)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(0, 0, 1)
    tempy = np.linspace(-env.y_max + env.y_max_init, env.y_max, per_side, endpoint=False)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(-env.x_max, env.x_max, per_side, endpoint=False)
    tempy = np.linspace(env.y_max, env.y_max, 1)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(-env.x_max, env.x_max, per_side, endpoint=False)
    tempy = np.linspace(-env.y_max, -env.y_max, 1)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    tempx = np.linspace(-env.x_max + env.x_max_init, env.x_max, per_side, endpoint=False)
    tempy = np.linspace(0, 0, 1)
    tx, ty = np.meshgrid(tempx, tempy)
    targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

    targets.add((env.x_max, env.y_max))

    targets = list(zip(*targets))

    env.x[env.n_robots:, 0] = targets[0]
    env.x[env.n_robots:, 1] = targets[1]


if __name__ == "__main__":
    # test generate_lattice() and reject_collisions()

    # triangular lattice
    lattice_vectors = [
        3. * np.array([-1.44, -1.44]),
        3. * np.array([-1.44, 1.44])]

    # square lattice
    # lattice_vectors = [
    #     np.array([-4.0, 0.]),
    #     np.array([0., -4.0])]

    free_region = (0, 100, 0, 100)
    spots = generate_lattice(free_region, lattice_vectors)

    obstacles = [(10, 45, 10, 90), (55, 90, 10, 90)]
    spots = reject_collisions(spots, obstacles)

    plt.figure()
    plt.plot([p[1] for p in spots], [p[0] for p in spots], '.')
    plt.show()


def generate_geometric_roads(n_cities, world_radius, road_radius):
    vertices = np.random.uniform(0.0, world_radius, size=(n_cities, 2))
    # edges, _ = _get_graph_edges(intercity_radius, cities, self_loops=True)

    # Build template graph on vertices using a Delauany triangulation and recover
    # the undirected edge list.
    tri = Delaunay(vertices)
    (indices, indptr) = tri.vertex_neighbor_vertices
    edges = []
    for i in range(vertices.shape[0]):
        for j in indptr[indices[i]:indices[i+1]]:
            if i < j:
                edges.append((i, j))
    # edge_list = np.array(edge_list)
    # edges = edge_list[edge_list[:,0] < edge_list[:,1]]

    extra_waypoints = []
    for (sender, receiver) in edges:
        p1 = np.reshape(vertices[sender, :], (1, 2))
        p2 = np.reshape(vertices[receiver, :], (1, 2))
        dist = np.linalg.norm(p1 - p2)
        n_new_points = int(dist/road_radius)
        extra_waypoints.extend([p1 + (p2-p1)/dist * n * road_radius for n in range(n_new_points)])
    all_waypoints = np.vstack([vertices, np.vstack(extra_waypoints)])
    return all_waypoints



