import math
import random
from typing import List
import numpy as np

import networkx as nx

c = 299792  # speed of liught in km/s


def hierarchy_pos(graph, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    graph: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(graph):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(iter(nx.topological_sort(
                graph)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(graph.nodes))

    def _hierarchy_pos(graph, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                       pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(graph.neighbors(root))
        if not isinstance(graph, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(graph, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)


def remove_duplicate_edges(lst):
    return [t for t in (set(tuple(i) for i in lst))]


def is_power_of_two(x):
    return x and (not (x & (x - 1)))


def randomise_spin_vector(spin_vector: List[float]) -> List[float]:
    """
    https://mathworld.wolfram.com/SpherePointPicking.html
    :param spin_vector:
    :return:
    """
    inital_mag = mag(spin_vector)

    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    w = np.random.normal(0, 1)
    norm = (u ** 2 + v ** 2 + w ** 2) ** (0.5)
    factor = inital_mag / norm
    new_spin_vector = [s * factor for s in [u, v, w]]
    final_mag = mag(new_spin_vector)
    np.testing.assert_almost_equal(final_mag, inital_mag, decimal=10,
                                   err_msg="mags {inital_mag} {final_mag} ")
    return new_spin_vector


def mag(x):
    return math.sqrt(math.fsum([i ** 2 for i in x]))
