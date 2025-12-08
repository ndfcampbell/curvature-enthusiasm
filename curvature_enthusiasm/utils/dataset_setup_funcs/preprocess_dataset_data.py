import os
import igl
import numpy as np
import trimesh
from trimesh.registration import procrustes
from curvature_enthusiasm.utils.mesh_utils import rescale_mesh_trimesh

# -------------------------- Core IO --------------------------

def _read_mesh(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh not found: {path}")
    v, f = igl.read_triangle_mesh(path)
    return v, f

def load_mesh_pair(dataset_name: str, source_pose: str, target_pose: str):
    """
    Assumes meshes live at:
        data/{dataset_name}/meshes/{pose}.obj
    (You've standardized this on disk.)
    """
    base = f"data/{dataset_name}/meshes"
    v_source_np, f_source_np = _read_mesh(os.path.join(base, f"{source_pose}.obj"))
    v_target_np, f_target_np = _read_mesh(os.path.join(base, f"{target_pose}.obj"))
    return v_source_np, f_source_np, v_target_np, f_target_np

# -------------------------- Core Utils --------------------------

# def apply_scale(V, F, scale_factor):
#     mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
#     mean_verts = np.mean(mesh.vertices, axis=0)
#     mesh.vertices -= mean_verts
#     mesh.apply_scale(scale_factor)
#     # mesh.vertices += mean_verts
#     return mesh.vertices


# def rescale_mesh_trimesh_raw(V, F, joint_positions=None, desired_volume=((4.0 * np.pi) / 3.0)):
#     # Create a mesh object
#     mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
#
#     if joint_positions is not None:
#         joint_positions -= np.mean(mesh.vertices, axis=0)
#
#     # Translate the mesh to be centered at the origin
#     mesh.vertices -= np.mean(mesh.vertices, axis=0)
#
#     volume = mesh.volume
#     # Compute the scale factor to adjust the volume to 1 (unit volume)
#     scale_factor = np.cbrt(desired_volume / volume)
#     # Rescale the mesh
#     mesh.apply_scale(scale_factor)
#
#     print(mesh.volume)
#
#     if joint_positions is not None:
#         joint_positions *= scale_factor
#
#     return mesh.vertices, joint_positions, scale_factor

# def rescale_mesh_trimesh(V, F, joint_positions=None, desired_volume=((4.0 * np.pi) / 3.0)):
#     # Create a mesh object
#     mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
#
#     # Save the original centroid
#     original_centroid = np.mean(mesh.vertices, axis=0)
#
#     if joint_positions is not None:
#         joint_positions -= original_centroid
#
#     # Translate the mesh to be centered at the origin
#     mesh.vertices -= original_centroid
#
#     volume = mesh.volume
#
#     print("Original Mesh Volume: ", mesh.volume)
#
#     # Compute the scale factor to adjust the volume to 1 (unit volume)
#     scale_factor = np.cbrt(desired_volume / volume)
#     # Rescale the mesh
#     mesh.apply_scale(scale_factor)
#
#     print("Rescaled Mesh Volume: ", mesh.volume)
#
#     if joint_positions is not None:
#         joint_positions *= scale_factor
#
#     # Translate back to original centroid
#     # mesh.vertices += original_centroid
#     # if joint_positions is not None:
#     #     joint_positions += original_centroid
#
#     return mesh.vertices, joint_positions


def calculate_center_of_gravity(vertices, faces, densities=None):
    """ Calculate the center of gravity of a 3D mesh using libigl for vertex area calculation.

    Parameters:
    vertices : numpy.ndarray
        Array of shape (N, 3) where N is the number of vertices.
        Each row represents the x, y, z coordinates of a vertex.
    faces : numpy.ndarray
        Array of shape (M, 3) where M is the number of faces.
        Each row contains the indices of vertices that form a triangular face.
    densities : numpy.ndarray, optional
        Array of shape (N,) representing the density or weight of each vertex.
        If not provided, assumes uniform density.

    Returns:
    numpy.ndarray
        1D array of shape (3,) representing the x, y, z coordinates of the center of gravity.
    """

    double_face_areas = igl.doublearea(vertices, faces)
    face_areas = double_face_areas / 2.0

    # Calculate vertex areas by distributing face areas to vertices
    vertex_areas = np.zeros(len(vertices))
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i] / 3.0

    if densities is not None:
        # If densities are provided, multiply them with vertex areas
        weights = vertex_areas * densities
    else:
        weights = vertex_areas

    # Calculate the total weight
    total_weight = np.sum(weights)

    # Calculate the weighted sum of positions
    weighted_sum = np.sum(vertices * weights[:, np.newaxis], axis=0)

    # Compute the center of gravity
    center_of_gravity = weighted_sum / total_weight

    return center_of_gravity






# def rescale_to_target_volume(mesh, target_volume):
#     """Return a scaled copy of the mesh with the desired volume."""
#     current_volume = abs(mesh.volume)
#     if current_volume == 0:
#         raise ValueError("Mesh volume is zero — likely not watertight.")
#     scale_factor = (target_volume / current_volume) ** (1 / 3)
#     mesh_scaled = mesh.copy()
#     mesh_scaled.apply_scale(scale_factor)
#
#     # Recheck result
#     new_volume = abs(mesh_scaled.volume)
#     print(f"Rescaled volume: {new_volume:.6f} (target: {target_volume:.6f})")
#
#     return mesh_scaled, scale_factor

# PROCRUSTES METHOD ROTATES AND TRANSLATES TO ALIGN
# def align_mesh(source_mesh, target_mesh):
#     """Align source to target using Procrustes (rigid transform only)."""
#     matrix, _, _ = procrustes(source_mesh.vertices, target_mesh.vertices, reflection=False, scale=False)
#     aligned_mesh = source_mesh.copy()
#     aligned_mesh.apply_transform(matrix)
#     return aligned_mesh, matrix

def align_mesh(source, target):
    """Compute translation that aligns centroids of source to target (no rotation)."""
    centroid_source = source.centroid
    centroid_target = target.centroid
    transform = np.eye(4)
    transform[:3, 3] = centroid_target - centroid_source
    return transform

def center_mesh(mesh):
    """Translate mesh so its centroid is at the origin."""
    mesh_centered = mesh.copy()
    centroid = mesh_centered.centroid
    mesh_centered.apply_translation(-centroid)
    return mesh_centered

def center_transform(mesh):
    """Return a transform that moves the mesh's centroid to the origin."""
    centroid = mesh.centroid
    transform = np.eye(4)
    transform[:3, 3] = -centroid
    return transform

def align_transform(source, target):
    """Compute rigid transform to align source to target (no scale)."""
    matrix, _, _ = procrustes(source.points, target.points, reflection=False, scale=False)
    return matrix


# def align_meshes(
#     v_source_np, f_source_np, v_target_np, f_target_np,
#     desired_volume=((4.0 * np.pi) / 3.0)):
#     """
#     Align two meshes using ONLY uniform scaling (to the same target volume)
#     and translation (to match centroids). No rotation is applied.
#     Returns:
#         vA_aligned, vB_aligned, scale_A, scale_B
#     """
#
#     mesh_A = trimesh.Trimesh(vertices=v_source_np, faces=f_source_np, process=False)
#     mesh_B = trimesh.Trimesh(vertices=v_target_np, faces=f_target_np, process=False)
#
#     # --- scale to target volume (uniform) ---
#     target_volume = desired_volume
#
#     def scale_to_volume(mesh, Vtgt):
#         # If volume<=0 (non-watertight), fall back to bbox volume as a heuristic
#         V = mesh.volume if mesh.is_volume else np.prod(mesh.extents)
#         if V <= 0:
#             V = np.prod(mesh.extents)
#         s = (Vtgt / V) ** (1.0 / 3.0)
#         scaled = mesh.copy()
#         scaled.apply_scale(s)
#         return scaled, s
#
#     mesh_A_scaled, scale_A = scale_to_volume(mesh_A, target_volume)
#     mesh_B_scaled, scale_B = scale_to_volume(mesh_B, target_volume)
#
#     # --- translate to match centroids, no rotation ---
#     cA = mesh_A_scaled.centroid
#     cB = mesh_B_scaled.centroid
#
#     # Center A at the origin
#     vA_centered = mesh_A_scaled.points - cA
#
#     # Move B so its centroid coincides with A's (which is at the origin now)
#     vB_centered = mesh_B_scaled.points - cB
#
#     # If you prefer A not centered at origin, add cA back to both:
#     vA_aligned = vA_centered  # A at origin
#     vB_aligned = vB_centered  # B translated to same origin
#
#     # Print volumes (should match target volume up to numeric tolerance)
#     # Note: volumes computed on meshes reconstructed from vertices to avoid stale caches
#     print(trimesh.Trimesh(vA_aligned, f_source_np, process=False).volume,
#           trimesh.Trimesh(vB_aligned, f_target_np, process=False).volume)
#
#     return vA_aligned, vB_aligned, scale_A, scale_B



def align_by_cog(source_v, source_f, target_v, target_f):
    """Translate target so its COG matches source COG."""
    c_s = calculate_center_of_gravity(source_v, source_f)
    c_t = calculate_center_of_gravity(target_v, target_f)
    return target_v - (c_t - c_s)

def align_by_min_bbox(source_v, target_v):
    """Translate target so its min corner matches source's min corner."""
    return target_v - (np.min(target_v, axis=0) - np.min(source_v, axis=0))

def _volume(v, f):
    """Translation-invariant volume for possibly open meshes: center first."""
    vc = v - v.mean(axis=0)          # center to kill origin-dependence
    return float(abs(trimesh.Trimesh(vertices=vc, faces=f, process=False).volume))

def debug_mesh_stats(v, f, label=""):
    vol = _volume(v, f)
    double_area = igl.doublearea(v, f)
    surface_area = np.sum(double_area) / 2.0
    print(f"[{label}] volume={vol:.6g}, sqrt_surface_area={np.sqrt(surface_area):.6g}")

def assert_volume(v, f, desired_volume, label, atol=1e-6):
    vol = _volume(v, f)
    if abs(vol - desired_volume) > atol:
        raise AssertionError(f"[{label}] volume {vol:.9f} != desired {desired_volume:.9f}")
    return vol



# ---------------- Bespoke preprocessors (examples) ------------


def preprocess_MANO(v_s, f_s, v_t, f_t, desired_volume):
    """
    Preprocess MANO meshes:
      1) Align target to source by center of gravity.
      2) Rescale both meshes independently to desired_volume.
      3) Apply extra correspondence-based offset (translation only).
      4) Verify volumes are still at desired_volume (translation invariant).
    """
    print("[MANO] preprocessing...")

    # 1) COG align (pure translation)
    v_t = align_by_cog(v_s, f_s, v_t, f_t)

    # 2) Rescale each to desired_volume
    v_s, _ = rescale_mesh_trimesh(v_s, f_s, desired_volume=desired_volume)
    v_t, _ = rescale_mesh_trimesh(v_t, f_t, desired_volume=desired_volume)

    # 3) Correspondence-based extra translation
    correspondence_vertices = [
        92, 234, 239, 279, 215, 214, 121, 78,
        79, 108, 120, 119, 117, 118, 122, 38
    ]
    offset = np.mean(v_t[correspondence_vertices] - v_s[correspondence_vertices], axis=0)
    v_t = v_t - offset

    debug_mesh_stats(v_s, f_s, "MANO source")
    debug_mesh_stats(v_t, f_t, "MANO target")

    return v_s, f_s, v_t, f_t



def preprocess_DFAUST(v_s, f_s, v_t, f_t, desired_volume):
    # Rescale both to desired volume
    v_s, _ = rescale_mesh_trimesh(v_s, f_s, desired_volume)
    v_t, _ = rescale_mesh_trimesh(v_t, f_t, desired_volume)

    # Align target to source by min bbox (matches your original)
    v_t = align_by_min_bbox(v_s, v_t)

    debug_mesh_stats(v_s, f_s, "DFAUST source")
    debug_mesh_stats(v_t, f_t, "DFAUST target")
    return v_s, f_s, v_t, f_t

def preprocess_FAUST(v_s, f_s, v_t, f_t, desired_volume):
    # COG align → rescale → COG align again (your original sequence)
    v_t = align_by_cog(v_s, f_s, v_t, f_t)
    v_s, _ = rescale_mesh_trimesh(v_s, f_s, desired_volume)
    v_t, _ = rescale_mesh_trimesh(v_t, f_t, desired_volume)
    v_t = align_by_cog(v_s, f_s, v_t, f_t)

    debug_mesh_stats(v_s, f_s, "FAUST source")
    debug_mesh_stats(v_t, f_t, "FAUST target")
    return v_s, f_s, v_t, f_t

def preprocess_SMAL(v_s, f_s, v_t, f_t, desired_volume):
    # Mean-offset removal, ICP-like alignment (you had align_source_mesh_to_target_mesh)
    # Here we keep a simple two-step approximation:
    v_t = v_t - np.mean(v_t - v_s, axis=0)
    # If you still have your custom alignment, call it here:
    # v_t = align_source_mesh_to_target_mesh(v_t, f_t, v_s, f_s)
    v_t = v_t - np.mean(v_t - v_s, axis=0)

    v_s, _ = rescale_mesh_trimesh(v_s, f_s, desired_volume)
    v_t, _ = rescale_mesh_trimesh(v_t, f_t, desired_volume)

    debug_mesh_stats(v_s, f_s, "SMAL source")
    debug_mesh_stats(v_t, f_t, "SMAL target")
    return v_s, f_s, v_t, f_t

def preprocess_TOPKIDS(v_s, f_s, v_t, f_t, desired_volume):
    """
    TOPKIDS: legacy behavior is a fixed scale of 1/20 for both meshes.
    (Joint scaling/loading should happen elsewhere, not in this mesh preprocessor.)
    """
    v_s = v_s / 20.0
    v_t = v_t / 20.0
    debug_mesh_stats(v_s, f_s, "TOPKIDS source")
    debug_mesh_stats(v_t, f_t, "TOPKIDS target")
    return v_s, f_s, v_t, f_t


def normalize_mesh_pair(v1, f1, v2, f2, desired_volume=1.0, use_convex_hull_if_open=True):
    """
    Normalize first mesh to desired volume centered at origin.
    Apply same transformation to second mesh.
    Prints volumes of both meshes before and after using existing debug utilities.

    Parameters:
    -----------
    v1, v2 : np.ndarray, shape (N, 3)
        Vertices of first and second mesh
    f1, f2 : np.ndarray, shape (F, 3)
        Faces of first and second mesh
    desired_volume : float
        Target volume for first mesh (default: 1.0)
    use_convex_hull_if_open : bool
        If True, use convex hull for open meshes (scale based on convex hull volume)

    Returns:
    --------
    v1_normalized, v2_normalized : np.ndarray
        Transformed vertices
    scale_factor : float
        Scale applied
    centroid : np.ndarray
        Translation applied
    """
    print("\n=== Standard Normalization ===")

    # Center at origin based on first mesh
    centroid = np.mean(v1, axis=0)
    v1_centered = v1 - centroid
    v2_centered = v2 - centroid

    # Calculate original volumes using existing helper
    print("Original meshes:")
    debug_mesh_stats(v1_centered, f1, "source")
    debug_mesh_stats(v2_centered, f2, "target")

    # Get volume for scaling calculation
    mesh1 = trimesh.Trimesh(vertices=v1_centered, faces=f1, process=False)
    is_watertight = mesh1.is_watertight

    if is_watertight:
        volume1_orig = abs(mesh1.volume)
        print(f"Source mesh is watertight, using exact volume")
    else:
        if use_convex_hull_if_open:
            # For open meshes with convex hull: we want the ACTUAL mesh to reach desired volume
            # So we compute scale based on actual volume, not convex hull
            volume1_orig = abs(mesh1.volume)
            print(f"Source mesh is open, using actual mesh volume for scaling")
            print(f"  (Convex hull volume: {abs(mesh1.convex_hull.volume):.6f})")
        else:
            bbox_size = np.ptp(v1_centered, axis=0)
            volume1_orig = np.prod(bbox_size)
            print(f"Source mesh is open, using bounding box volume")

    # Calculate scale factor: volume_new = scale^3 * volume_old
    scale_factor = (desired_volume / volume1_orig) ** (1 / 3)
    print(f"\nTarget volume: {desired_volume:.6f}")
    print(f"Scale factor: {scale_factor:.6f}")

    # Apply transformation
    v1_normalized = v1_centered * scale_factor
    v2_normalized = v2_centered * scale_factor

    # Verify normalized volumes using existing helper
    print("\nNormalized meshes:")
    debug_mesh_stats(v1_normalized, f1, "source")
    debug_mesh_stats(v2_normalized, f2, "target")

    # Calculate and print volume ratio
    volume1_norm = _volume(v1_normalized, f1)
    volume2_norm = _volume(v2_normalized, f2)
    print(f"Volume ratio (target/source): "
          f"before={_volume(v2_centered, f2) / volume1_orig:.6f}, "
          f"after={volume2_norm / volume1_norm:.6f}")

    return v1_normalized, v2_normalized, scale_factor, centroid


# --------------- Registry and dispatcher ---------------------

_PREPROCESS_REGISTRY = {
    "MANO":       preprocess_MANO,
    "DFAUST":     preprocess_DFAUST,
    "SMAL":       preprocess_SMAL,
    "TOPKIDS":    preprocess_TOPKIDS,
}

def preprocess_mesh_pair(dataset_name, v_source_np, f_source_np, v_target_np, f_target_np, desired_volume=((4.0 * np.pi) / 3.0)):
    """
    If a bespoke preprocessor exists for dataset_name, apply it.
    Otherwise, return inputs unchanged.
    """
    func = _PREPROCESS_REGISTRY.get(dataset_name)

    # Apply bespoke preprocessing if it exists
    if func is not None:
        v_source_norm, f_source_np, v_target_norm, f_target_np = func(v_source_np, f_source_np, v_target_np, f_target_np, desired_volume)

        # THIS IS A HACK, AS WE DON'T HAVE NORMALIZATION INFO FOR THE DATASETS
        norm = {
            'scale': None,
            'centroid': None
        }

    else:
        # Otherwise, normalize to desired volume
        print(f"No bespoke preprocessor for '{dataset_name}', applying standard normalization...")
        v_source_norm, v_target_norm, scale, centroid = normalize_mesh_pair(
            v_source_np,f_source_np, v_target_np, f_target_np,
            desired_volume=desired_volume
        )

        norm = {
            'scale': scale,
            'centroid': centroid
        }

    return v_source_norm, f_source_np, v_target_norm, f_target_np, norm



# def align_meshes(v_source_np, f_source_np, v_target_np, f_target_np, desired_volume=((4.0 * np.pi) / 3.0)):
#
#     mesh_A = trimesh.Trimesh(vertices=v_source_np, faces=f_source_np, process=False)
#     mesh_B = trimesh.Trimesh(vertices=v_target_np, faces=f_target_np, process=False)
#
#
#     # Compute target volume
#     # if not mesh_A.is_watertight or not mesh_B.is_watertight:
#     #     raise ValueError("Meshes A and B must be watertight to compute volume.")
#     target_volume = desired_volume
#
#     # Rescale A and B
#     mesh_A_scaled, scale_A = rescale_to_target_volume(mesh_A, target_volume)
#     mesh_B_scaled, scale_B = rescale_to_target_volume(mesh_B, target_volume)
#
#     # Get transforms
#     T_center_B = center_transform(mesh_B_scaled)
#     mesh_B_centered = mesh_B_scaled.copy()
#     mesh_B_centered.apply_transform(T_center_B)
#
#     T_center_A = center_transform(mesh_A_scaled)
#     mesh_A_centered = mesh_A_scaled.copy()
#     mesh_A_centered.apply_transform(T_center_A)
#
#     T_align = align_transform(mesh_B_centered, mesh_A_centered)
#     mesh_B_aligned = mesh_B_centered.copy()
#     mesh_B_aligned.apply_transform(T_align)
#
#     # Apply the SAME chain of transforms to C: scale, T_center_B, T_align
#     # print volumes
#     print(mesh_A_centered.volume, mesh_B_aligned.volume)
#
#     return mesh_A_centered.vertices, mesh_B_aligned.vertices, scale_A, scale_B
