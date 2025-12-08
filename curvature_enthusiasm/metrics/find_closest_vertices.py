from sklearn.neighbors import NearestNeighbors

def find_closest_vertices(X, Y):
    """
    For each vertex in X, find the closest vertex in Y.

    Parameters:
    X (np.array): Array of shape (n, 3) representing vertices of the first mesh.
    Y (np.array): Array of shape (m, 3) representing vertices of the second mesh.

    Returns:
    np.array: Array of shape (n,) containing indices of the closest vertices in Y for each vertex in X.
    """
    # Create a NearestNeighbors object
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y)

    # Find the nearest neighbor for each point in X
    distances, indices = nn.kneighbors(X)

    # The indices array is of shape (n, 1), so we flatten it to (n,)
    return indices.flatten()