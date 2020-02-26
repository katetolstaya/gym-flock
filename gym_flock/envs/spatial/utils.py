import numpy as np


def _get_graph_edges(rad, pos1, pos2=None, self_loops=False):
    """
    Get list of edges from agents in positions pos1 to positions pos2.
    for agents closer than distance rad
    :param rad: "communication" radius
    :param pos1: first set of positions
    :param pos2: second set of positions
    :param self_loops: boolean flag indicating whether to include self loops
    :return: (senders, receivers), edge features
    """
    r = np.linalg.norm(_get_pos_diff(pos1, pos2), axis=2)
    r[r > rad] = 0
    if not self_loops and pos2 is None:
        np.fill_diagonal(r, 0)
    edges = np.nonzero(r)
    return edges, r[edges]


def _get_pos_diff(sender_loc, receiver_loc=None):
    """
    Get matrix of distances between agents in positions pos1 to positions pos2.
    :param sender_loc: first set of positions
    :param receiver_loc: second set of positions (use sender_loc if None)
    :return: matrix of distances, len(pos1) x len(pos2)
    """
    n = sender_loc.shape[0]
    m = sender_loc.shape[1]
    if receiver_loc is not None:
        n2 = receiver_loc.shape[0]
        m2 = receiver_loc.shape[1]
        diff = sender_loc.reshape((n, 1, m)) - receiver_loc.reshape((1, n2, m2))
    else:
        diff = sender_loc.reshape((n, 1, m)) - sender_loc.reshape((1, n, m))
    return diff


def _get_k_edges(k, pos1, pos2=None, self_loops=False, allow_nearest=False):
    """
    Get list of edges from agents in positions pos1 to closest agents in positions pos2.
    Each agent in pos1 will have K outgoing edges.
    :param k: number of edges
    :param pos1: first set of positions
    :param pos2: second set of positions
    :param self_loops: boolean flag indicating whether to include self loops
    :param allow_nearest: allow the nearest landmark as an action or remove it
    :return: (senders, receivers), edge features
    """
    r = np.linalg.norm(_get_pos_diff(pos1, pos2), axis=2)

    if not self_loops and pos2 is None:
        np.fill_diagonal(r, np.Inf)

    mask = np.zeros(np.shape(r))
    if allow_nearest:
        idx = np.argpartition(r, k - 1, axis=1)[:, 0:k]
        mask[np.arange(np.shape(pos1)[0])[:, None], idx] = 1
    else:  # remove the closest edge
        idx = np.argpartition(r, k, axis=1)[:, 0:k + 1]
        mask[np.arange(np.shape(pos1)[0])[:, None], idx] = 1
        idx = np.argmin(r, axis=1)
        mask[np.arange(np.shape(pos1)[0])[:], idx] = 0

    edges = np.nonzero(mask)
    return edges, r[edges]


