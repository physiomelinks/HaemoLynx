import sys
import re

with open("examples/carotid_image_to_model.py", "r") as f:
    content = f.read()

# Replace _load_and_preprocess_image definition
old_def = "def _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, graph_config, vis_config, pipeline_config, args=None):"
if old_def not in content:
    # Try finding without args
    old_def = "def _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, graph_config, vis_config, pipeline_config):"

new_def = """def _load_raw_probability_field(image_path, input_format, pre_config, skel_config):
    # Load the 3D or 4D volume using lazy loading to save memory
    if input_format == "tif":
        image = io.load_3d_tif(image_path, lazy=True)
    elif input_format == "h5":
        if not H5_DATASET_NAME:
            raise ValueError("Set H5_DATASET_NAME when INPUT_FORMAT is 'h5'.")
        image = io.load_3d_h5(image_path, H5_DATASET_NAME, lazy=True)
    else:
        raise ValueError("INPUT_FORMAT must be 'tif' or 'h5'.")

    is_lazy = preprocessing.image._is_dask_array(image)
    if is_lazy:
        import dask.array as da
        logger.info("Using Dask for lazy out-of-core preprocessing.")

    print(f"Loaded image shape: {image.shape}")

    if 0 < skel_config.sub_volume_percentage < 1.0 or skel_config.sub_volume_offset_z != 0 or \\
       skel_config.sub_volume_offset_y != 0 or skel_config.sub_volume_offset_x != 0:
        print(f"Applying ROI crop (sub-volume={skel_config.sub_volume_percentage})...")
        image = preprocessing.crop_roi(
            image,
            sub_volume_percentage=skel_config.sub_volume_percentage,
            offset_z=skel_config.sub_volume_offset_z,
            offset_y=skel_config.sub_volume_offset_y,
            offset_x=skel_config.sub_volume_offset_x
        )
        print(f"  ROI new shape: {image.shape}")

    entropy_map = None
    if image.ndim == 4:
        if pre_config.enable_shannon_entropy:
            entropy_map = preprocessing.calculate_entropy_map(image)
        dims = np.array(image.shape)
        c_axis = np.argmin(dims)
        if c_axis == 0: image = image[pre_config.ilastik_vessel_channel, :, :, :]
        elif c_axis == 1: image = image[:, pre_config.ilastik_vessel_channel, :, :]
        elif c_axis == 2: image = image[:, :, pre_config.ilastik_vessel_channel, :]
        else: image = image[:, :, :, pre_config.ilastik_vessel_channel]
        if entropy_map is not None and is_lazy:
            entropy_map = entropy_map.compute()

    if is_lazy:
        raw_prob_map = image.compute()
    else:
        raw_prob_map = image.copy()
        
    return raw_prob_map, entropy_map

def _preprocess_local_mask(raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config, optimize_trials=0, optimize_patience=15):
    if optimize_trials > 0:
        import ImageLynx.statistics.benchmarking as benchmarking
        import ImageLynx.statistics.auto_tuner as auto_tuner
        import copy
        def pre_eval_callback(suggested_kwargs):
            test_config_dict = pre_config.__dict__.copy()
            test_config_dict.update(suggested_kwargs)
            _, test_binary = _apply_preprocessing_filters(
                raw_prob_map, entropy_map, test_config_dict, 
                boundary_permeability_mode=graph_config.boundary_permeability_mode
            )
            return benchmarking.run_all_preprocessing_benchmarks(raw_prob_map, test_binary, entropy_map)
        best_pre_params = auto_tuner.run_optuna_preprocessing_optimization(
            pre_eval_callback, n_trials=optimize_trials,
            output_dir=pipeline_config.vtk_output_prefix.parent,
            patience=optimize_patience
        )
        for k, v in best_pre_params.items():
            setattr(pre_config, k, v)

    filtered_image, binary = _apply_preprocessing_filters(
        raw_prob_map, entropy_map, pre_config.__dict__,
        boundary_permeability_mode=graph_config.boundary_permeability_mode
    )

    if skel_config.closing_radius > 0:
        binary = preprocessing.skeleton.close_binary_mask(binary, radius=skel_config.closing_radius)
    if skel_config.bridge_gap_size > 0:
        binary = preprocessing.skeleton.bridge_gaps(binary, max_gap=skel_config.bridge_gap_size)
    if skel_config.prune_mask_before > 0:
        binary = preprocessing.skeleton.keep_largest_mask_components(
            binary, n_components=skel_config.prune_mask_before, connectivity=skel_config.component_connectivity
        )
    return raw_prob_map, binary

def old_marker():"""

# Find the start of _load_and_preprocess_image
start_idx = content.find(old_def)
if start_idx != -1:
    # Find the end of the function (return image, binary) or (return (image.compute()...))
    # It ends with: "return (image.compute() if preprocessing.image._is_dask_array(image) else image), binary"
    end_marker = "return (image.compute() if preprocessing.image._is_dask_array(image) else image), binary"
    end_idx = content.find(end_marker, start_idx)
    if end_idx != -1:
        end_idx += len(end_marker)
        # Also remove any trailing newlines inside the function
        content = content[:start_idx] + new_def + content[end_idx:]

with open("examples/carotid_image_to_model.py", "w") as f:
    f.write(content)
