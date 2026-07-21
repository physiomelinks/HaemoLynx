import sys

with open("examples/carotid_image_to_model.py", "r") as f:
    content = f.read()

old_func = """def _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, graph_config, vis_config, pipeline_config):
    \"\"\"
    Phase 1: Loads the image, handles 4D channels/entropy, crops the ROI,
    removes noise, and applies hysteresis thresholding to generate a binary mask.
    \"\"\""""
    
start_idx = content.find(old_func)
if start_idx == -1:
    print("Could not find _load_and_preprocess_image")
    sys.exit(1)

end_marker = "return (image.compute() if preprocessing.image._is_dask_array(image) else image), binary"
end_idx = content.find(end_marker, start_idx) + len(end_marker)

new_funcs = """def _apply_preprocessing_filters(image, entropy_map, pre_config_dict, boundary_permeability_mode):
    # --- Virtual Padding (Boundary Caging Fix) ---
    pad_z, pad_y, pad_x = 0, 0, 0
    if boundary_permeability_mode == "caged":
        pad_z = 10
    elif boundary_permeability_mode in ["universal_sink", "robin_resistance"]:
        pad_z, pad_y, pad_x = 10, 10, 10

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        image = np.pad(image, pad_width=((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), mode='edge')

    import ImageLynx.preprocessing as preprocessing
    median_size = pre_config_dict.get("median_filter_size", 0)
    if median_size > 0:
        image = preprocessing.median_filter_image(image, size=median_size)

    opening_radius = pre_config_dict.get("morphological_opening_radius", 0)
    if opening_radius > 0:
        image = preprocessing.morphological_opening(image, radius=opening_radius)

    closing_radius = pre_config_dict.get("morphological_closing_radius", 0)
    if closing_radius > 0:
        image = preprocessing.morphological_closing(image, radius=closing_radius)

    if pre_config_dict.get("probability_smoothing_sigma", 0) > 0:
        image = preprocessing.smooth_probability_map(image, sigma=pre_config_dict["probability_smoothing_sigma"])

    if pre_config_dict.get("enable_hysteresis_threshold", True):
        binary = preprocessing.hysteresis_threshold(
            image,
            low=pre_config_dict.get("hysteresis_threshold_low", 0.2),
            high=pre_config_dict.get("hysteresis_threshold_high", 0.4)
        )
    else:
        from skimage.filters import threshold_otsu
        binary = image > threshold_otsu(image)

    if pre_config_dict.get("enable_hole_filling", True):
        binary = preprocessing.skeleton.fill_holes_3d(binary)

    # --- Remove Virtual Padding ---
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        z_slice = slice(pad_z, -pad_z) if pad_z > 0 else slice(None)
        y_slice = slice(pad_y, -pad_y) if pad_y > 0 else slice(None)
        x_slice = slice(pad_x, -pad_x) if pad_x > 0 else slice(None)
        image = image[z_slice, y_slice, x_slice]
        binary = binary[z_slice, y_slice, x_slice]
        
    return image, binary

def _load_raw_probability_field(image_path, input_format, pre_config, skel_config):
    import ImageLynx.io as io
    import ImageLynx.preprocessing as preprocessing
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)

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
    import ImageLynx.preprocessing as preprocessing
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
    return filtered_image, binary"""

# We need to replace the old block with the new_funcs.
old_apply_filters = "def _apply_preprocessing_filters("
if old_apply_filters in content:
    apply_start = content.find(old_apply_filters)
    content = content[:apply_start] + new_funcs + content[end_idx:]
else:
    content = content[:start_idx] + new_funcs + content[end_idx:]

main_exec_old = """    if pipeline_config.do_skeletonize:
        image, binary = _load_and_preprocess_image(image_path, input_format, pre_config, skel_config, graph_config, vis_config, pipeline_config)
        
        # --- Optuna Hyperparameter Optimization ---"""

main_exec_new = """    if pipeline_config.do_skeletonize:
        raw_prob_map, entropy_map = _load_raw_probability_field(image_path, input_format, pre_config, skel_config)
        
        if getattr(pipeline_config, 'chunk_fraction', None) is not None and pipeline_config.chunk_fraction < 1.0:
            if getattr(pipeline_config, 'export_grid_preview', False):
                import sys
                import pyvista as pv
                from ImageLynx.graph.tiling import generate_evenly_distributed_bounding_boxes
                import numpy as np
                import ImageLynx.io as io
                
                print(f"\\n--- Generating Map-Reduce Grid Preview (fraction={pipeline_config.chunk_fraction}) ---")
                spacing = io.get_tif_spacing(image_path) if input_format == "tif" else (1.0, 1.0, 1.0)
                
                grid_mask = np.zeros(raw_prob_map.shape, dtype=np.uint8)
                
                for bbox in generate_evenly_distributed_bounding_boxes(raw_prob_map.shape, pipeline_config.chunk_fraction, margin=pipeline_config.margin):
                    z1, z2, y1, y2, x1, x2 = bbox['core']
                    
                    if z1 < raw_prob_map.shape[0]: grid_mask[z1, y1:y2, x1:x2] = 255
                    if z2 - 1 >= 0 and z2 - 1 < raw_prob_map.shape[0]: grid_mask[z2-1, y1:y2, x1:x2] = 255
                    if y1 < raw_prob_map.shape[1]: grid_mask[z1:z2, y1, x1:x2] = 255
                    if y2 - 1 >= 0 and y2 - 1 < raw_prob_map.shape[1]: grid_mask[z1:z2, y2-1, x1:x2] = 255
                    if x1 < raw_prob_map.shape[2]: grid_mask[z1:z2, y1:y2, x1] = 255
                    if x2 - 1 >= 0 and x2 - 1 < raw_prob_map.shape[2]: grid_mask[z1:z2, y1:y2, x2-1] = 255
                
                vtk_vol = pv.ImageData()
                vtk_vol.dimensions = np.array(raw_prob_map.shape)
                vtk_vol.spacing = (spacing[2], spacing[1], spacing[0])
                
                vtk_vol.point_data["Probability"] = raw_prob_map.flatten(order="F").astype(np.float32)
                vtk_vol.point_data["ChunkGrid"] = grid_mask.flatten(order="F")
                
                out_path = pipeline_config.vtk_output_prefix.with_name(f"{pipeline_config.vtk_output_prefix.name}_grid_preview.vti")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                vtk_vol.save(out_path)
                
                print(f"Exported Grid Preview to: {out_path}")
                print("Exiting pipeline early as requested.")
                sys.exit(0)

        image, binary = _preprocess_local_mask(
            raw_prob_map, entropy_map, pre_config, skel_config, graph_config, pipeline_config,
            optimize_trials=args.optimize_preprocessing, optimize_patience=pipeline_config.optimize_patience
        )
        
        # --- Optuna Hyperparameter Optimization ---"""

content = content.replace(main_exec_old, main_exec_new)

with open("examples/carotid_image_to_model.py", "w") as f:
    f.write(content)
