"""Model creation functions."""
import equinox as eqx
from .model import CE_MODEL, IK_System_DQ, NODE, GSO_MLP

def create_sampling_settings(bone_sampling_config):
    """Create sampling settings dictionary from config."""
    radii = bone_sampling_config.axis_radii
    # Extract all radii for each bone in sorted order
    axis_radii = [radii[key] for key in sorted(radii.keys())]

    return {
        'N_RIGID_SAMPLES_PER_BONE': bone_sampling_config.n_rigid_samples_per_bone,
        'N_TISSUE_SAMPLES_PER_ITER': bone_sampling_config.n_tissue_samples_per_iter,
        'AXIS_RADII': axis_radii,
    }

def create_model(ik_template_skeleton, bone_sampling_config, node_config,
                 ode_random_key, quat_random_key,
                 raw_pred_quat_key, nn_dtype, var_dtype):
    """Create the main ARC_PLUS model with all components."""

    # Create sampling settings
    sampling_settings = create_sampling_settings(bone_sampling_config)

    # Create IK system
    ik_system = IK_System_DQ(
        ik_template_skeleton,
        sampling_settings,
        dtype=nn_dtype
    )

    # Create ODE function
    ode_func = NODE(
        input_size=4,
        output_size=3,
        width_size=256,
        depth=6,
        activation_func=node_config.activation_fn,
        out_scale=1e-1,
        dtype=nn_dtype,
        key=ode_random_key
    )


    # Create rotation network
    rot_net = GSO_MLP(
        in_size=4,
        width=128,
        depth=3,
        dtype=nn_dtype,
        key=quat_random_key
    )

    # Create main model
    model = CE_MODEL(
        ik_system=ik_system,
        ode_func=ode_func,
        conformal_func=rot_net,
        n_tissue_samples=bone_sampling_config.n_tissue_samples_per_iter,
        n_ode_timesteps=node_config.n_ode_steps,
        key=raw_pred_quat_key,
        dtype=var_dtype
    )

    return model

def load_model_from_file(training_data, mesh_data):
    """Load a pretrained model from file."""
    config = training_data['config']

    model = create_model(
        ik_template_skeleton=mesh_data['ik_template_skeleton'],
        bone_sampling_config=config.bone_sampling,
        node_config=config.node,
        ode_random_key=training_data['ode_random_key'],
        quat_random_key=training_data['quat_random_key'],
        raw_pred_quat_key=training_data['raw_pred_quat_key'],
        nn_dtype=training_data['nn_dtype'],
        var_dtype=training_data['var_dtype']
    )

    model = eqx.tree_deserialise_leaves(training_data['model_fn'], model)
    return model