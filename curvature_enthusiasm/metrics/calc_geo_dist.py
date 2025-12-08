import networkx as nx
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors


def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = 500

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), 'Graph not connected'
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increase NN.')
    return geodesic_x