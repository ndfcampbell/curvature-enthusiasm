import time
import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from curvature_enthusiasm.metrics.chamfer_loss import chamfer_dist


def show_interpolation_animation(
        dataset,
        source,
        target,
        frame_dir='animation',
        start_frame=0,
        end_frame=99,
        ext='.obj',
        base_fps=30,
        compute_chamfer=True,
        chamfer_cmap_range=(0.0, 0.5),
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_to_show = np.arange(start_frame, end_frame + 1, 1)
    edge_width = 0.3

    # --- Colors from reference image ---
    colors = {
        'target': (0.55, 0.70, 0.60),  # Muted green (left figure)
        'source': (0.85, 0.80, 0.65),  # Warm beige/tan (right figure)
        'prediction': (0.85, 0.80, 0.65),  # Warm beige/tan (right figure)
        'animation': (0.45, 0.60, 0.75),  # Flat blue for deforming mesh
    }

    # --- Paths ---
    data_dir = f'results/{dataset}/{dataset}_{source}-{target}/'

    # --- Load meshes ---
    V_in, F_in = igl.read_triangle_mesh(data_dir + 'source' + ext)
    V_out, F_out = igl.read_triangle_mesh(data_dir + 'target' + ext)
    V_pred_final, F = igl.read_triangle_mesh(data_dir + f'{frame_dir}/frame_{frames_to_show[-1]}{ext}')

    if compute_chamfer:
        V_pred_tensor = torch.from_numpy(V_pred_final).to(device).float()
        V_out_tensor = torch.from_numpy(V_out).to(device).float()

        with torch.no_grad():  # Disable gradient computation for memory savings
            chamfer = chamfer_dist(V_pred_tensor, V_out_tensor, full=True, return_numpy=True)

        del V_pred_tensor, V_out_tensor

        mean_chamfer = np.mean(chamfer)
        print(f"Chamfer distance: {mean_chamfer:.4f}")
    else:
        chamfer = None

    # --- Load all frames ---
    v_list = []
    for f in frames_to_show:
        V, _ = igl.read_triangle_mesh(data_dir + f'{frame_dir}/frame_{f}{ext}')
        v_list.append(V)

    n_frames = len(v_list)

    # --- Init Polyscope with Modern Settings ---
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("y_up")
    ps.set_front_dir("y_front")
    ps.set_ground_plane_height_factor(0.0)  # Flatter ground
    ps.set_shadow_darkness(0.15)  # Lighter shadows
    ps.set_shadow_blur_iters(15)  # Softer shadows
    ps.set_background_color((0.98, 0.98, 0.99))  # Very light background

    # --- Register static meshes with modern material ---
    ps.register_surface_mesh(
        "mesh_in", V_in, F_in,
        material='clay',  # Modern smooth material
        smooth_shade=True,
        color=colors['source'],
        edge_width=edge_width
    )

    ps.register_surface_mesh(
        "mesh_out", V_out, F_out,
        material='clay',
        smooth_shade=True,
        color=colors['target'],
        edge_width=edge_width
    )

    ps.register_surface_mesh(
        "mesh_pred", V_pred_final, F,
        material='clay',
        smooth_shade=True,
        color=colors['prediction'],
        edge_width=edge_width
    )

    ps.load_color_map("white_yellow_red_black", "visualisations/custom_gradient_colormap.png")

    if compute_chamfer and chamfer is not None:
        ps.get_surface_mesh("mesh_pred").add_scalar_quantity(
            "chamfer distance",
            chamfer,
            enabled=True,
            cmap="white_yellow_red_black",
            vminmax=chamfer_cmap_range
        )

    # --- Register animation mesh ---
    ps_mesh_frames = ps.register_surface_mesh(
        'output', v_list[0], F,
        material='clay',
        smooth_shade=True,
        color=colors['animation'],
        enabled=True,
        edge_width=edge_width
    )

    # --- Animation State ---
    frame_idx = 0
    playing = False
    speed_multiplier = 1.0
    frame_interval = 1.0 / (base_fps * speed_multiplier)
    last_time = time.time()

    def callback():
        nonlocal frame_idx, playing, speed_multiplier, frame_interval, last_time

        # --- Speed Slider ---
        psim.Text("Playback speed:")
        changed_speed, new_speed = psim.SliderFloat("##speed_slider", float(speed_multiplier), 0.1, 2.0, "%.1f")
        if changed_speed:
            speed_multiplier = new_speed
            frame_interval = 1.0 / (base_fps * speed_multiplier)
            print(f"[UI] Speed multiplier set to {speed_multiplier:.2f}")

        # --- Frame Navigation ---
        changed_idx, new_idx = psim.InputInt("Frame", frame_idx, step=1, step_fast=5)
        if changed_idx and 0 <= new_idx < n_frames:
            frame_idx = new_idx
            ps_mesh_frames.update_vertex_positions(v_list[frame_idx])

        # --- Play / Pause Buttons ---
        if psim.Button("Play"):
            playing = True
        psim.SameLine()
        if psim.Button("Pause"):
            playing = False

        # --- Animation Update ---
        current_time = time.time()
        if playing and (current_time - last_time) >= frame_interval:
            frame_idx = (frame_idx + 1) % n_frames
            ps_mesh_frames.update_vertex_positions(v_list[frame_idx])
            last_time = current_time

    # --- Launch UI ---
    ps.set_user_callback(callback)
    ps.show()


if __name__ == '__main__':
    show_interpolation_animation(
        dataset='MANO',
        source='01_01r',
        target='01_02r',
        frame_dir='animation',
        start_frame=0,
        end_frame=99,
        compute_chamfer=True
    )

