from typing import Tuple, List
import numpy as np

# For NTU RGB+D, assume node 21 (centre of chest)
# is the "centre of gravity" mentioned in the paper

num_nodes = 25
epsilon = 1e-6

# Directed edges: (source, target), see
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf
# for node IDs, and reduce index to 0-based
directed_edges = [(i-1, j-1) for i, j in [
    (1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
    (6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
    (10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
    (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
    (21, 2), (21, 3), (21, 5), (21, 9)
]]


def normalize_incidence_matrix(im):
    im /= (im.sum(-1)+epsilon)[:, np.newaxis]
    return im


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
    max_edges = len(edges)
    source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
    for edge_id, (source_node, target_node) in enumerate(edges):
        source_graph[source_node, edge_id] = 1.
        target_graph[target_node, edge_id] = 1.
    source_graph = normalize_incidence_matrix(source_graph)
    target_graph = normalize_incidence_matrix(target_graph)
    return source_graph, target_graph


class Graph:
    def __init__(self):
        super().__init__()
        self.num_nodes = num_nodes
        self.edges = directed_edges
        # Incidence matrices
        self.source_M, self.target_M = \
            build_digraph_incidence_matrix(self.num_nodes, self.edges)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    graph = Graph()
    source_M = graph.source_M
    target_M = graph.target_M

    plt.imshow(source_M)
    plt.show()

    plt.imshow(target_M)
    plt.show()
