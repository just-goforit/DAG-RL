import sys
sys.path.append('..')
import queue
import numpy as np

def get_TopLevel(n:int, edge_index:np.ndarray, start_from:int=0):
    """
    get Top level

    parameters
    ----------
    * n: node count, id from 0-n-1

    * edge_index: sparse edge index

    * start_from: level start from
 
    example
    -------
    >>> n = 5
    >>> edge_index = [[0, 1, 2, 3], 
    >>>               [3, 3, 4, 4]]

    return 
    ------
    >>> [0, 0, 0, 1, 2]
    """
    def _get_successors(node_id, edge_index):
        start_index = np.searchsorted(edge_index[:, 0], node_id, side='left')
        end_index = np.searchsorted(edge_index[:, 0], node_id, side='right')
        successors = edge_index[start_index:end_index, 1]
        return successors

    level = [0 for _ in range(n)]
    in_degree = [0 for _ in range(n)]

    for i in range(len(edge_index[0])):
        in_degree[edge_index[1, i]] += 1

    edge_index = edge_index.T.copy()
    edge_index = edge_index[edge_index[:, 0].argsort()]

    q = queue.Queue(maxsize=n)
    for i in range(n):
        if in_degree[i] == 0:
            q.put(i)
    lv = start_from
    sep = -1
    while not q.empty():
        node = q.get()
        if sep == node:
            lv += 1
            sep = -1
        level[node] = lv
        successors = _get_successors(node, edge_index)
        if len(successors) > 0:
            for successor in np.nditer(successors):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    q.put(successor)
                    if sep == -1:
                        sep = successor
    return np.array(level)   
