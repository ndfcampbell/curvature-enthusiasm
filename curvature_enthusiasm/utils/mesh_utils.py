from typing import Tuple, Optional, Union

import numpy as np
import open3d as o3d
import trimesh
import igl
from plyfile import PlyData, PlyElement

from jaxtyping import Array, Float, Int


def has_bdry(f):
    """Returns True if mesh has a boundary."""
    return len(igl.boundary_loop(f)) > 0

def rescale_mesh_trimesh(
        V: Float[Array, "V 3"],
        F: Int[Array, "M 3"],
        joint_positions: Optional[Float[Array, "J 3"]] = None,
        desired_volume: float = (4.0 * np.pi) / 3.0
) -> Tuple[Float[Array, "N 3"], Optional[Float[Array, "J 3"]]]:
    """
    Rescale a mesh to have a desired volume, optionally rescaling joint positions.

    Args:
        V: Vertex positions array of shape (n_vertices, 3)
        F: Face indices array of shape (n_faces, 3)
        joint_positions: Optional joint positions array of shape (n_joints, 3)
        desired_volume: Target volume for the rescaled mesh

    Returns:
        Tuple of:
        - Rescaled vertex positions of shape (n_vertices, 3)
        - Rescaled joint positions of shape (n_joints, 3) or None
    """

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    if joint_positions is not None:
        joint_positions -= np.mean(mesh.vertices, axis=0)

    # Translate the mesh to be centered at the origin
    mesh.vertices -= np.mean(mesh.vertices, axis=0)

    volume = mesh.volume
    # Compute the scale factor to adjust the volume to 1 (unit volume)
    scale_factor = np.cbrt(desired_volume / volume)
    # Rescale the mesh
    mesh.apply_scale(scale_factor)

    print(mesh.volume)

    if joint_positions is not None:
        joint_positions *= scale_factor

    return mesh.vertices, joint_positions


def subdivide(
        uv: Float[Array, "N 3"],
        uf: Int[Array, "M 3"],
        levels: int = 1
) -> Tuple[Float[Array, "N2 3"], Int[Array, "M2 3"]]:
    """
    Subdivide a mesh using Loop subdivision.

    Args:
        uv: Vertex positions array of shape (n_vertices, 3)
        uf: Face indices array of shape (n_faces, 3)
        levels: Number of subdivision levels to apply

    Returns:
        Tuple of:
        - Subdivided vertex positions of shape (n_vertices_new, 3)
        - Subdivided face indices of shape (n_faces_new, 3)
    """
    print(f'Subdividing mesh {levels} time(s)...')

    # Make copies to avoid modifying inputs
    uv = uv.copy()
    uf = uf.copy()

    for n in range(levels):
        uv, uf = igl.upsample(uv, uf)
        print(f'  Level {n + 1}: {len(uv)} vertices, {len(uf)} faces')

    return uv, uf

def calculate_vertex_area_proportions(
    vertices: Float[Array, "N 3"],
    faces: Int[Array, "M 3"],
) -> Float[Array, " N"]:
    """
    Calculate the proportion of total mesh area associated with each vertex.

    For each vertex, this computes the sum of 1/3 of the areas of all incident
    triangles, then normalizes by the total mesh area.

    Args:
        vertices: Vertex positions array of shape (n_vertices, 3)
        faces: Face indices array of shape (n_faces, 3)

    Returns:
        Array of shape (n_vertices,) containing the area proportion for each vertex.
        Values sum to 1.0.
    """

    # Calculate the areas of all triangles in the mesh using libigl
    triangle_areas = igl.doublearea(vertices, faces) / 2.0

    # Initialize an array to store the area associated with each vertex
    vertex_areas = np.zeros(len(vertices))

    # Iterate through all faces (triangles) in the mesh
    for face, area in zip(faces, triangle_areas):
        # Distribute the face area equally among its three vertices
        vertex_areas[face] += area / 3.0

    # Calculate the total surface area of the mesh
    total_area = np.sum(triangle_areas)

    # Calculate the proportion of total area for each vertex
    vertex_proportions = vertex_areas / total_area

    return vertex_proportions

def procrustes_align(source_points, target_points):
    """
    Align source points to target points using Procrustes analysis.
    """
    # Center both point sets
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute the covariance matrix
    covariance_matrix = np.dot(target_centered.T, source_centered)

    # Compute the optimal rotation using SVD
    U, _, Vt = np.linalg.svd(covariance_matrix)
    rotation = np.dot(U, Vt)

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(U, Vt)

    # Compute the scale factor
    scale = np.sum(target_centered**2) / np.sum(source_centered**2)
    scale = np.sqrt(scale)

    # Compute the translation
    translation = target_centroid - scale * np.dot(source_centroid, rotation.T)

    return scale, rotation, translation

def align_meshes(source_mesh, target_mesh):
    """
    Align source mesh to target mesh using Procrustes method.
    """
    # Get vertices from meshes
    source_points = np.asarray(source_mesh.points)
    target_points = np.asarray(target_mesh.points)

    # Perform Procrustes alignment
    scale, rotation, translation = procrustes_align(source_points, target_points)

    # Transform the source mesh
    transformed_mesh = source_mesh.transform(np.eye(4))  # Create a copy
    transformed_mesh.scale(scale, center=np.zeros(3))
    transformed_mesh.rotate(rotation, center=np.zeros(3))
    transformed_mesh.translate(translation)

    return transformed_mesh


def create_mesh_from_vertices_and_faces(vertices, faces):
    """
    Create an Open3D mesh from vertices and faces.

    :param vertices: numpy array of shape (N, 3) containing vertex coordinates
    :param faces: numpy array of shape (M, 3) containing face indices
    :return: Open3D TriangleMesh object
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def align_source_mesh_to_target_mesh(source_verts, source_faces, target_verts, target_faces):
    """
    Align source mesh to target mesh using Procrustes method.
    """

    source_mesh = create_mesh_from_vertices_and_faces(source_verts, source_faces)
    target_mesh = create_mesh_from_vertices_and_faces(target_verts, target_faces)
    pred_yt_o3d = align_meshes(source_mesh, target_mesh)
    aligned_verts = np.asarray(pred_yt_o3d.points)
    return aligned_verts





def export_points_to_ply(test_points, filename):
    """
    Export points to PLY file (no faces).

    Parameters:
    -----------
    test_points : array, shape (K, 3, 3)
        K triangles, each with 3 vertices of 3D coordinates
    filename : str
        Output PLY filename
    """
    # Flatten to point cloud
    points = test_points.reshape(-1, 3)  # Shape: (K*3, 3)

    # Create structured array for plyfile
    vertex_data = np.array([tuple(p) for p in points],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Create PlyElement
    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    # Write to file
    PlyData([vertex_element]).write(filename)
