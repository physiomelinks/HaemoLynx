#!/usr/bin/env python3
"""ImageLynx main pipeline package."""
import logging
import sys
import inspect
import ast
import pickle
import json
from pathlib import Path
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Ensure package and sibling example modules are importable.
root_dir = Path(__file__).resolve().parents[1]
examples_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.haemodynamics.pipeline import HaemodynamicsApplyConfig, apply_poiseuille_haemodynamics
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz
from ImageLynx.parsers import add_schema_arguments, cli_overrides, dump_config, load_config
from preflight import run_preflight_checklist
from resistance_pipeline_schema import SCHEMA
from resistance_pipeline_settings import *  # noqa: F403
from wizard import run_interactive_setup_wizard


def run_pipeline_stages(image_path=INPUT_PATH,
                            use_ilastik_segmentation=USE_ILASTIK_SEGMENTATION,
                            ilastik_unsegmented_image_path=ILASTIK_UNSEGMENTED_IMAGE_PATH,
                            ilastik_classifier_path=ILASTIK_CLASSIFIER_PATH,
                            ilastik_executable=ILASTIK_EXECUTABLE,
                            ilastik_output_dir=ILASTIK_OUTPUT_DIR,
                            ilastik_output_suffix=ILASTIK_OUTPUT_SUFFIX,
                            voxel_size_override_xyz=VOXEL_SIZE_OVERRIDE_XYZ,
                            voxel_size_policy=VOXEL_SIZE_POLICY,
                            axis_order=IMAGE_AXIS_ORDER,
                            use_large_vessel_masks=USE_LARGE_VESSEL_MASKS,
                            use_ilastik_large_vessel_segmentation=USE_ILASTIK_LARGE_VESSEL_SEGMENTATION,
                            large_vessel_mask_dilation_microns=LARGE_VESSEL_MASK_DILATION_MICRONS,
                            use_small_vessel_masks_for_boundary_assignment=USE_SMALL_VESSEL_MASKS_FOR_BOUNDARY_ASSIGNMENT,
                            use_ilastik_small_vessel_segmentation=USE_ILASTIK_SMALL_VESSEL_SEGMENTATION,
                            small_vessel_mask_min_overlap_fraction=SMALL_VESSEL_MASK_MIN_OVERLAP_FRACTION,
                            write_small_vessel_boundary_labelling_3d_html=WRITE_SMALL_VESSEL_BOUNDARY_LABELLING_3D_HTML,
                            automated_vessel_assignment=AUTOMATED_VESSEL_ASSIGNMENT,
                            large_arteriole_mask_path=LARGE_ARTERIOLE_MASK_PATH,
                            large_venule_mask_path=LARGE_VENULE_MASK_PATH,
                            small_arteriole_mask_path=SMALL_ARTERIOLE_MASK_PATH,
                            small_venule_mask_path=SMALL_VENULE_MASK_PATH,
                            ilastik_unsegmented_arteriole_image_path=ILASTIK_UNSEGMENTED_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_venule_image_path=ILASTIK_UNSEGMENTED_VENULE_IMAGE_PATH,
                            ilastik_arteriole_classifier_path=ILASTIK_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_venule_classifier_path=ILASTIK_VENULE_CLASSIFIER_PATH,
                            ilastik_unsegmented_small_arteriole_image_path=ILASTIK_UNSEGMENTED_SMALL_ARTERIOLE_IMAGE_PATH,
                            ilastik_unsegmented_small_venule_image_path=ILASTIK_UNSEGMENTED_SMALL_VENULE_IMAGE_PATH,
                            ilastik_small_arteriole_classifier_path=ILASTIK_SMALL_ARTERIOLE_CLASSIFIER_PATH,
                            ilastik_small_venule_classifier_path=ILASTIK_SMALL_VENULE_CLASSIFIER_PATH,
                            diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
                            constriction_by_branch_order=CONSTRICTION_BY_BRANCH_ORDER,
                            do_pericyte_constriction=DO_PERICYTE_CONSTRUCTION,
                            use_pericyte_mask_constriction=USE_PERICYTE_MASK_CONSTRICTION,
                            pericyte_mask_path=PERICYTE_MASK_PATH,
                            pericyte_mask_h5_dataset_name=PERICYTE_MASK_H5_DATASET_NAME,
                            pericyte_max_assignment_distance_um=PERICYTE_MAX_ASSIGNMENT_DISTANCE_UM,
                            pericyte_min_diameter_um=PERICYTE_MIN_DIAMETER_UM,
                            pericyte_max_diameter_um=PERICYTE_MAX_DIAMETER_UM,
                            use_probabilistic_pericyte_constriction=USE_PROBABILISTIC_PERICYTE_CONSTRICTION,
                            pericyte_constriction_probability=PERICYTE_CONSTRICTION_PROBABILITY,
                            run_pericyte_resistance_comparison=RUN_PERICYTE_RESISTANCE_COMPARISON,
                            pericyte_comparison_baseline_value=PERICYTE_COMPARISON_BASELINE_VALUE,
                            pericyte_comparison_constricted_value=PERICYTE_COMPARISON_CONSTRICTED_VALUE,
                            reuse_comparison_pericyte_cohort_for_main_run=REUSE_COMPARISON_PERICYTE_COHORT_FOR_MAIN_RUN,
                            plot_dir=BASE_PLOT_DIR,
                            verbose_logging=VERBOSE_LOGGING,
                            do_skeletonize=DO_SKELETONIZE,
                            do_graph_building=DO_GRAPH_BUILDING,
                            run_haemodynamics=RUN_HAEMODYNAMICS,
                            do_equiv_resistance_calculation=DO_EQUIV_RESISTANCE_CALCULATION,
                            min_branch_length=MIN_BRANCH_LENGTH,
                            min_stub_length=MIN_STUB_LENGTH,
                            cluster_collapse_distance=CLUSTER_COLLAPSE_DISTANCE,
                            vtk_output_prefix=VTK_OUTPUT_PREFIX,
                            skeleton_closing_radius=SKELETON_CLOSING_RADIUS,
                            skeleton_bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
                            skeleton_min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
                            skeleton_max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
                            skeleton_component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
                            skeleton_min_component_percent=SKELETON_MIN_COMPONENT_PERCENT,
                            graph_reconnect_threshold=GRAPH_RECONNECT_THRESHOLD,
                            final_orphan_reconnect_threshold=FINAL_ORPHAN_RECONNECT_THRESHOLD,
                            starting_node_selection_method=STARTING_NODE_SELECTION_METHOD,
                            output_node_selection_method=OUTPUT_NODE_SELECTION_METHOD,
                            arteriole_boundary_selection_method=ARTERIOLE_BOUNDARY_SELECTION_METHOD,
                            venule_boundary_selection_method=VENULE_BOUNDARY_SELECTION_METHOD,
                            starting_node_coordinates=STARTING_NODE_COORDINATES,
                            output_node_coordinates=OUTPUT_NODE_COORDINATES,
                            arteriole_boundary_node_coordinates=ARTERIOLE_BOUNDARY_NODE_COORDINATES,
                            venule_boundary_node_coordinates=VENULE_BOUNDARY_NODE_COORDINATES,
                            starting_node_volumes=STARTING_NODE_VOLUMES,
                            output_node_volumes=OUTPUT_NODE_VOLUMES,
                            arteriole_boundary_node_volumes=ARTERIOLE_BOUNDARY_NODE_VOLUMES,
                            venule_boundary_node_volumes=VENULE_BOUNDARY_NODE_VOLUMES,
                            strict_branch_order_assignment=STRICT_BRANCH_ORDER_ASSIGNMENT,
                            starting_nodes=STARTING_NODES, 
                            output_nodes=OUTPUT_NODES, 
                            arteriole_boundary_nodes=ARTERIOLE_BOUNDARY_NODES,
                            venule_boundary_nodes=VENULE_BOUNDARY_NODES,
                            input_p_bc=INPUT_P_BC, 
                            output_p_bc=OUTPUT_P_BC, 
                            visualize_results=VISUALIZE_RESULTS, 
                            interactive_plots=INTERACTIVE_PLOTS,
                            show_plots_in_ide=SHOW_PLOTS_IN_IDE,
                            ide_plot_mode=IDE_PLOT_MODE,
                            hold_ide_plots_open=HOLD_IDE_PLOTS_OPEN,
                            final_render_mode=FINAL_RENDER_MODE,
                            visualize_vtk=VISUALIZE_VTK,
                            measurement_3d_to_cell_mask=MEASUREMENT_3D_TO_CELL_MASK,
                            cell_mask_path=CELL_MASK_PATH,
                            cell_mask_h5_dataset_name=CELL_MASK_H5_DATASET_NAME,
                            measurement_3d_vessel_mask_path=MEASUREMENT_3D_VESSEL_MASK_PATH,
                            measurement_3d_vessel_mask_h5_dataset_name=MEASUREMENT_3D_VESSEL_MASK_H5_DATASET_NAME,
                            measurement_3d_reference_image_path=MEASUREMENT_3D_REFERENCE_IMAGE_PATH,
                            measurement_3d_reference_h5_dataset_name=MEASUREMENT_3D_REFERENCE_H5_DATASET_NAME,
                            statistics_mode=STATISTICS_MODE,
                            use_fwhm_edge_diameters=USE_FWHM_EDGE_DIAMETERS,
                            fwhm_raw_tiff_path=FWHM_RAW_TIFF_PATH,
                            fwhm_sample_spacing_along_edge_um=FWHM_SAMPLE_SPACING_ALONG_EDGE_UM,
                            fwhm_transverse_profile_step_um=FWHM_TRANSVERSE_PROFILE_STEP_UM,
                            fwhm_transverse_half_extent_um=FWHM_TRANSVERSE_HALF_EXTENT_UM,
                            fwhm_diameter_guess_um=FWHM_DIAMETER_GUESS_UM,
                            fwhm_min_total_extent_multiplier=FWHM_MIN_TOTAL_EXTENT_MULTIPLIER,
                            fwhm_background_label=FWHM_BACKGROUND_LABEL,
                            fwhm_junction_label=FWHM_JUNCTION_LABEL,
                            fwhm_allow_junction_crossing=FWHM_ALLOW_JUNCTION_CROSSING,
                            fwhm_profile_baseline_mode=FWHM_PROFILE_BASELINE_MODE,
                            fwhm_profile_baseline_wing_fraction=FWHM_PROFILE_BASELINE_WING_FRACTION,
                            fwhm_constrain_fitted_baseline=FWHM_CONSTRAIN_FITTED_BASELINE,
                            fwhm_baseline_constraint_half_width_ptp=FWHM_BASELINE_CONSTRAINT_HALF_WIDTH_PTP,
                            fwhm_clip_profile_to_single_vessel=FWHM_CLIP_PROFILE_TO_SINGLE_VESSEL,
                            fwhm_clip_min_drop_fraction_of_center=FWHM_CLIP_MIN_DROP_FRACTION_OF_CENTER,
                            fwhm_clip_re_rise_fraction_of_center=FWHM_CLIP_RE_RISE_FRACTION_OF_CENTER,
                            fwhm_branch_endpoint_exclusion_um=FWHM_BRANCH_ENDPOINT_EXCLUSION_UM,
                            fwhm_junction_proximity_exclusion_um=FWHM_JUNCTION_PROXIMITY_EXCLUSION_UM,
                            fwhm_enforce_same_edge_locality=FWHM_ENFORCE_SAME_EDGE_LOCALITY,
                            fwhm_same_edge_arc_window_um=FWHM_SAME_EDGE_ARC_WINDOW_UM,
                            fwhm_same_edge_arc_window_multiplier=FWHM_SAME_EDGE_ARC_WINDOW_MULTIPLIER,
                            fwhm_same_edge_arc_window_min_um=FWHM_SAME_EDGE_ARC_WINDOW_MIN_UM,
                            fwhm_cap_half_extent_by_nonlocal_same_edge_distance=FWHM_CAP_HALF_EXTENT_BY_NONLOCAL_SAME_EDGE_DISTANCE,
                            fwhm_nonlocal_same_edge_arc_separation_um=FWHM_NONLOCAL_SAME_EDGE_ARC_SEPARATION_UM,
                            fwhm_nonlocal_same_edge_half_extent_factor=FWHM_NONLOCAL_SAME_EDGE_HALF_EXTENT_FACTOR,
                            fwhm_reject_samples_with_center_offset=FWHM_REJECT_SAMPLES_WITH_CENTER_OFFSET,
                            fwhm_max_fit_center_offset_um=FWHM_MAX_FIT_CENTER_OFFSET_UM,
                            fwhm_reject_samples_with_low_fit_r2=FWHM_REJECT_SAMPLES_WITH_LOW_FIT_R2,
                            fwhm_min_fit_r2=FWHM_MIN_FIT_R2) -> None:
    image_path = Path(image_path)
    if use_ilastik_segmentation:
        unsegmented_image_path = Path(ilastik_unsegmented_image_path)
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if ilastik_classifier_path is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        ilastik_output_dir = Path(ilastik_output_dir)
        ilastik_segmented_path = ilastik_output_dir / (
            f"{unsegmented_image_path.stem}_segmented{ilastik_output_suffix}"
        )
        print(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        image_path = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(ilastik_classifier_path),
            output_path=ilastik_segmented_path,
            ilastik_executable=ilastik_executable,
        )
        print(f"Using ilastik-segmented image: {image_path}")
    else:
        print(f"Using segmented input image: {image_path}")

    image_path = io.resolve_image_path_with_optional_zip(image_path)
    # get image format from image_path
    input_format = image_path.suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    axis_order = io.normalize_axis_order(axis_order, label="IMAGE_AXIS_ORDER")
    if axis_order != io.CANONICAL_AXIS_ORDER:
        print(
            f"Input axis order '{axis_order}' will be transposed to the canonical "
            f"'{io.CANONICAL_AXIS_ORDER}' layout on load."
        )
    vtk_output_prefix = Path(vtk_output_prefix)
    output_dir = vtk_output_prefix.parent
    valid_final_render_modes = {"2d", "3d"}
    if final_render_mode not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{final_render_mode}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose_logging else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{image_path.stem}_skeleton.npy"
    voxel_meta_path = output_dir / f"{image_path.stem}_voxel_size.json"
    graph_path = output_dir / f"{image_path.stem}_graph.pkl"
    projection_path = plot_dir / "skeleton_projection.png"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    if do_skeletonize:
        if input_format in {"tif", "tiff"}:
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_tif(
                image_path,
                axis_order=axis_order,
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        elif input_format == "h5":
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_h5(
                image_path,
                axis_order=axis_order,
            )
            metadata_voxel_size = (
                float(voxel_size_x),
                float(voxel_size_y),
                float(voxel_size_z),
            )
        else:
            raise ValueError("INPUT_FORMAT must be 'tif', 'tiff', or 'h5'.")
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
        )
        print(
            "Voxel-size resolution: "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"metadata={metadata_voxel_size}, "
            f"final={voxel_size}"
        )
        
        preprocessing.print_skeleton_connectivity_stats(
            "raw",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        visualization.visualize_skeleton(skeleton, save_path=plot_dir / "raw_skeleton.png")

        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            min_branch_length=skeleton_min_branch_length,
            max_bridge_distance=skeleton_max_bridge_distance,
            component_connectivity=skeleton_component_connectivity,
            min_component_fraction=skeleton_min_component_percent / 100.0,
            closing_radius=skeleton_closing_radius,
            bridge_gap_size=skeleton_bridge_gap_size,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=skeleton_component_connectivity,
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
        voxel_meta_path.write_text(
            json.dumps(
                {
                    "voxel_size": voxel_size,
                    "voxel_size_source": voxel_size_source,
                    "voxel_metadata_status": voxel_meta_status,
                    "voxel_size_policy": voxel_size_policy,
                    "voxel_size_override_xyz": voxel_size_override_xyz,
                }
            )
        )
        print(f"Saved skeleton to: {skeleton_path}")
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = io.apply_axis_order(tifffile.imread(image_path), axis_order)
        if voxel_meta_path.exists():
            cached_voxel_meta = json.loads(voxel_meta_path.read_text())
            metadata_voxel_size = tuple(cached_voxel_meta["voxel_size"])
            voxel_meta_status = cached_voxel_meta.get(
                "voxel_metadata_status",
                {"source": "cache", "status": "unknown"},
            )
        else:
            metadata_voxel_size = (1.0, 1.0, 1.0)
            voxel_meta_status = {"source": "none", "status": "missing"}
        voxel_size, voxel_size_source = resolve_voxel_size_xyz(
            metadata_voxel_size_xyz=metadata_voxel_size,
            metadata_status=voxel_meta_status,
            voxel_size_override_xyz=voxel_size_override_xyz,
            voxel_size_policy=voxel_size_policy,
        )
        print(f"Loaded skeleton from: {skeleton_path}")
        print(
            "Voxel-size resolution (from cache/default): "
            f"source={voxel_size_source}, "
            f"metadata_status={voxel_meta_status.get('status')}, "
            f"final={voxel_size}"
        )

    print("Visualizing skeleton projection...")
    visualization.visualize_skeleton(skeleton, save_path=projection_path)
    print("Skeleton projection saved.")

    main_voxel_size_xyz = tuple(float(v) for v in voxel_size)
    # Image metadata reports (x, y, z); array axes are canonical (z, y, x). Everything
    # downstream that scales array indices uses voxel_size_zyx.
    voxel_size_zyx = io.voxel_size_zyx_from_xyz(main_voxel_size_xyz)
    (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_mask_voxel_size,
        large_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        mask_role="large",
        enabled=use_large_vessel_masks,
        use_ilastik=use_ilastik_large_vessel_segmentation,
        arteriole_mask_path=large_arteriole_mask_path,
        venule_mask_path=large_venule_mask_path,
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
        ilastik_unsegmented_arteriole_path=ilastik_unsegmented_arteriole_image_path,
        ilastik_unsegmented_venule_path=ilastik_unsegmented_venule_image_path,
        ilastik_arteriole_classifier_path=ilastik_arteriole_classifier_path,
        ilastik_venule_classifier_path=ilastik_venule_classifier_path,
        ilastik_output_dir=ilastik_output_dir,
        ilastik_output_suffix=ilastik_output_suffix,
        ilastik_executable=ilastik_executable,
        dilation_microns=large_vessel_mask_dilation_microns,
        axis_order=axis_order,
    )
    (
        small_arteriole_mask,
        small_venule_mask,
        small_arteriole_mask_voxel_size,
        small_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        mask_role="small",
        enabled=use_small_vessel_masks_for_boundary_assignment,
        use_ilastik=use_ilastik_small_vessel_segmentation,
        arteriole_mask_path=small_arteriole_mask_path,
        venule_mask_path=small_venule_mask_path,
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
        ilastik_unsegmented_arteriole_path=ilastik_unsegmented_small_arteriole_image_path,
        ilastik_unsegmented_venule_path=ilastik_unsegmented_small_venule_image_path,
        ilastik_arteriole_classifier_path=ilastik_small_arteriole_classifier_path,
        ilastik_venule_classifier_path=ilastik_small_venule_classifier_path,
        ilastik_output_dir=ilastik_output_dir,
        ilastik_output_suffix=ilastik_output_suffix,
        ilastik_executable=ilastik_executable,
        axis_order=axis_order,
        loaded_message_suffix=(
            f"min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f}"
        ),
    )

    if do_graph_building:
        # 3) Convert skeleton to graph.
        def _graph_build_step_callback(graph_obj, label: str) -> None:
            plot_png = label
            if label == "smart_multigraph_degree2_removal_pass1":
                plot_png = "smart_multigraph_degree2_removal"
            visualization.save_graph_snapshot(
                graph_obj,
                image,
                output_dir,
                plot_dir,
                image_path.stem,
                label,
            )
            visualization.visualize_edges_and_nodes(
                image,
                graph_obj,
                label_nodes=True,
                save_path=plot_dir / f"{plot_png}.png",
            )

        G = graph.build_graph_from_skeleton(
            skeleton,
            voxel_size=voxel_size_zyx,
            graph_reconnect_threshold=graph_reconnect_threshold,
            final_orphan_reconnect_threshold=final_orphan_reconnect_threshold,
            cluster_collapse_distance=cluster_collapse_distance,
            min_stub_length=min_stub_length,
            debug=verbose_logging,
            step_callback=_graph_build_step_callback,
        )

        with graph_path.open("wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
    else:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found at {graph_path}. "
                "Set DO_GRAPH_BUILDING=True to generate it first."
            )
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph from: {graph_path}")

    # Store physical voxel-unit metadata used for skeleton/graph geometry and mask alignment.
    G.graph["image_voxel_size_xyz"] = main_voxel_size_xyz
    G.graph["image_voxel_size_zyx"] = voxel_size_zyx
    if large_arteriole_mask is not None and large_venule_mask is not None:
        G.graph["large_arteriole_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_arteriole_mask_voxel_size
        )
        G.graph["large_venule_mask_voxel_size_xyz"] = tuple(
            float(v) for v in large_venule_mask_voxel_size
        )
    
    # Visualize final graph used for boundary-node verification.
    if final_render_mode == "3d":
        final_graph_3d_path = plot_dir / "final_graph_3d.html"
        visualization.visualize_3d_plotly(
            G,
            title="Final Graph (Interactive 3D)",
            save_html_path=str(final_graph_3d_path),
            show=show_plots_in_ide or interactive_plots,
        )
        print(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
    else:
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=False,
            save_path=plot_dir / "final_graph.png",
            show_coordinates_degree_1=True,
        )

    auto_start_nodes: list[int] = []
    auto_output_nodes: list[int] = []
    if automated_vessel_assignment:
        if large_arteriole_mask is None or large_venule_mask is None:
            raise ValueError(
                "automated_vessel_assignment=True requires arteriole and venule masks. "
                "Set use_large_vessel_masks=True and provide mask paths."
            )
        auto_start_nodes, auto_output_nodes = (
            graph.select_terminal_nodes_from_large_vessel_masks(
                G,
                large_arteriole_mask=large_arteriole_mask,
                large_venule_mask=large_venule_mask,
                voxel_size_zyx=voxel_size_zyx,
                allow_overlap=False,
            )
        )
        if not auto_start_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "arteriole mask (after any configured dilation)."
            )
        if not auto_output_nodes:
            raise ValueError(
                "automated_vessel_assignment=True found no terminal nodes in the "
                "venule mask (after any configured dilation)."
            )
        starting_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_start_nodes
        ]
        output_node_coordinates = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_output_nodes
        ]
        automated_assignment_html_path = plot_dir / "automated_vessel_assignment_3d.html"
        wrote_assignment_html = graph.write_automated_vessel_assignment_3d_html(
            G,
            large_arteriole_mask=large_arteriole_mask,
            large_venule_mask=large_venule_mask,
            input_nodes=auto_start_nodes,
            output_nodes=auto_output_nodes,
            voxel_size_zyx=voxel_size_zyx,
            output_html_path=automated_assignment_html_path,
        )
        if wrote_assignment_html:
            print(
                "Saved automated vessel-assignment 3D visualization to: "
                f"{automated_assignment_html_path}"
            )
        else:
            print(
                "Skipped automated vessel-assignment 3D visualization "
                "(plotly is not installed)."
            )
        print(
            "Automated vessel assignment selected "
            f"{len(starting_node_coordinates)} input coordinates from arteriole-mask overlap "
            f"and {len(output_node_coordinates)} output coordinates from venule-mask overlap."
        )

    starting_nodes[:] = []
    output_nodes[:] = []
    arteriole_boundary_nodes[:] = []
    venule_boundary_nodes[:] = []
    if automated_vessel_assignment:
        # Use direct terminal-node overlap assignment from vessel masks.
        start_nodes = auto_start_nodes
        out_nodes = [node_id for node_id in auto_output_nodes if node_id not in set(start_nodes)]
    else:
        start_nodes = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=starting_node_selection_method,
            node_role="input",
            coordinates=starting_node_coordinates,
            volume_boxes=starting_node_volumes,
        )
        out_nodes = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=output_node_selection_method,
            node_role="output",
            coordinates=output_node_coordinates,
            volume_boxes=output_node_volumes,
            exclude_nodes=start_nodes,
        )
    starting_nodes.extend(start_nodes)
    output_nodes.extend(out_nodes)
    used_nodes = set(starting_nodes) | set(output_nodes)
    if arteriole_boundary_node_coordinates or arteriole_boundary_node_volumes:
        art_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=arteriole_boundary_selection_method,
            node_role="input",
            coordinates=arteriole_boundary_node_coordinates,
            volume_boxes=arteriole_boundary_node_volumes,
            exclude_nodes=list(used_nodes),
        )
        arteriole_boundary_nodes.extend(art_boundary)
        used_nodes.update(arteriole_boundary_nodes)

    if venule_boundary_node_coordinates or venule_boundary_node_volumes:
        ven_boundary = graph.select_boundary_nodes_by_method(
            G,
            image.shape,
            method=venule_boundary_selection_method,
            node_role="output",
            coordinates=venule_boundary_node_coordinates,
            volume_boxes=venule_boundary_node_volumes,
            exclude_nodes=list(used_nodes),
        )
        venule_boundary_nodes.extend(ven_boundary)
    if use_small_vessel_masks_for_boundary_assignment:
        if small_arteriole_mask is None or small_venule_mask is None:
            raise ValueError(
                "use_small_vessel_masks_for_boundary_assignment=True requires "
                "small_arteriole_mask_path and small_venule_mask_path."
            )
        inferred_boundary_results = graph.infer_boundary_nodes_from_small_vessel_masks(
            G,
            small_arteriole_mask=small_arteriole_mask,
            small_venule_mask=small_venule_mask,
            voxel_size_zyx=voxel_size_zyx,
            minimum_overlap_fraction=float(small_vessel_mask_min_overlap_fraction),
            allow_overlap=False,
        )
        arteriole_boundary_nodes[:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        venule_boundary_nodes[:] = list(inferred_boundary_results["venule_boundary_nodes"])
        print(
            "Small-vessel mask boundary assignment selected "
            f"{len(arteriole_boundary_nodes)} arteriole boundary nodes and "
            f"{len(venule_boundary_nodes)} venule boundary nodes "
            f"(min_overlap_fraction={float(small_vessel_mask_min_overlap_fraction):.3f})."
        )
        print(
            "Small-vessel mask edge labels: "
            f"arteriole_edges={inferred_boundary_results['arteriole_edge_count']}, "
            f"venule_edges={inferred_boundary_results['venule_edge_count']}, "
            f"overlap_edges={inferred_boundary_results['overlap_edge_count']}."
        )
        if write_small_vessel_boundary_labelling_3d_html:
            boundary_html = Path(plot_dir) / "small_vessel_mask_boundary_labelling_3d.html"
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            ok = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                G,
                small_arteriole_mask=small_arteriole_mask,
                small_venule_mask=small_venule_mask,
                arteriole_boundary_nodes=arteriole_boundary_nodes,
                venule_boundary_nodes=venule_boundary_nodes,
                voxel_size_zyx=voxel_size_zyx,
                output_html_path=boundary_html,
            )
            if ok:
                print(f"Saved interactive 3D small-vessel boundary view: {boundary_html}")
            else:
                print(
                    "Small-vessel boundary 3D HTML not written (install plotly to enable)."
                )
    if automated_vessel_assignment:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES directly from terminal-node overlap with vessel masks."
        )
    else:
        print(
            f"Selected {len(starting_nodes)} STARTING_NODES and {len(output_nodes)} "
            "OUTPUT_NODES from manual coordinates."
        )
    print(f"Starting nodes are: {starting_nodes}")
    print(f"Output nodes are: {output_nodes}")
    print(f"Arteriole boundary nodes are: {arteriole_boundary_nodes}")
    print(f"Venule boundary nodes are: {venule_boundary_nodes}")

    if starting_nodes and output_nodes:
        resistance_node_pair = (starting_nodes[0], output_nodes[0])
        print(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        if automated_vessel_assignment:
            raise ValueError(
                "No starting or output nodes found from terminal-node overlap with "
                "arteriole/venule masks."
            )
        raise ValueError(
            "No starting or output nodes found from manual input coordinates."
        )

    # 4) Add branch orders and hemodynamic edge weights.
    if starting_nodes:

        def _vessel_types_after_branch_assign(graph_obj) -> None:
            vessel_type_3d_path = plot_dir / "vessel_types_assigned_3d.html"
            visualization.visualize_3d_plotly_vessel_types(
                graph_obj,
                title="Assigned Vessel Types (Interactive 3D)",
                save_html_path=str(vessel_type_3d_path),
                show=False,
            )
            print(
                "Saved vessel-type 3D visualization after branch assignment to: "
                f"{vessel_type_3d_path}"
            )

        branch_summary = graph.assign_vessel_branch_orders(
            G,
            starting_nodes,
            output_nodes=output_nodes,
            arteriole_boundary_nodes=arteriole_boundary_nodes,
            venule_boundary_nodes=venule_boundary_nodes,
            strict_hierarchical=strict_branch_order_assignment,
            expects_hierarchical=bool(
                automated_vessel_assignment
                or use_small_vessel_masks_for_boundary_assignment
            ),
            post_assign_callback=_vessel_types_after_branch_assign,
        )
        if branch_summary["mode"] == "hierarchical":
            print(
                "Assigned hierarchical branch orders "
                "(Art*/Ven* first, then capillary B* from arteriole boundary)."
            )
            print(f"Branch assignment summary: {branch_summary}")
        elif branch_summary["mode"] == "capillary":
            print(
                "Assigned capillary branch orders from STARTING_NODES only "
                "(no arteriole/venule boundary-node sets supplied)."
            )

        if not run_haemodynamics:
            print(
                "Haemodynamics disabled; skipping diameter fitting and "
                "Poiseuille conductance assignment."
            )
        elif run_haemodynamics:
            haemo_config = HaemodynamicsApplyConfig(
                diameter_by_branch_order=diameter_by_branch_order,
                constriction_by_branch_order=constriction_by_branch_order,
                custom_edges=custom_edges,
                do_pericyte_constriction=do_pericyte_constriction,
                use_pericyte_mask_constriction=use_pericyte_mask_constriction,
                pericyte_mask_path=pericyte_mask_path,
                pericyte_mask_h5_dataset_name=pericyte_mask_h5_dataset_name,
                pericyte_max_assignment_distance_um=pericyte_max_assignment_distance_um,
                pericyte_min_diameter_um=pericyte_min_diameter_um,
                pericyte_max_diameter_um=pericyte_max_diameter_um,
                use_probabilistic_pericyte_constriction=use_probabilistic_pericyte_constriction,
                pericyte_constriction_probability=pericyte_constriction_probability,
                run_pericyte_resistance_comparison=run_pericyte_resistance_comparison,
                pericyte_comparison_baseline_value=pericyte_comparison_baseline_value,
                pericyte_comparison_constricted_value=pericyte_comparison_constricted_value,
                reuse_comparison_pericyte_cohort_for_main_run=reuse_comparison_pericyte_cohort_for_main_run,
                comparison_output_csv_path=(
                    output_dir / f"{image_path.stem}_pericyte_resistance_comparison.csv"
                    if run_pericyte_resistance_comparison
                    else None
                ),
                resistance_node_pair=resistance_node_pair,
                use_fwhm_edge_diameters=use_fwhm_edge_diameters,
                fwhm_raw_tiff_path=fwhm_raw_tiff_path,
                voxel_size_zyx=voxel_size_zyx,
                axis_order=axis_order,
                fwhm_sample_spacing_along_edge_um=fwhm_sample_spacing_along_edge_um,
                fwhm_transverse_profile_step_um=fwhm_transverse_profile_step_um,
                fwhm_transverse_half_extent_um=fwhm_transverse_half_extent_um,
                fwhm_diameter_guess_um=fwhm_diameter_guess_um,
                fwhm_min_total_extent_multiplier=fwhm_min_total_extent_multiplier,
                fwhm_background_label=fwhm_background_label,
                fwhm_junction_label=fwhm_junction_label,
                fwhm_allow_junction_crossing=fwhm_allow_junction_crossing,
                fwhm_profile_baseline_mode=fwhm_profile_baseline_mode,
                fwhm_profile_baseline_wing_fraction=fwhm_profile_baseline_wing_fraction,
                fwhm_constrain_fitted_baseline=fwhm_constrain_fitted_baseline,
                fwhm_baseline_constraint_half_width_ptp=fwhm_baseline_constraint_half_width_ptp,
                fwhm_clip_profile_to_single_vessel=fwhm_clip_profile_to_single_vessel,
                fwhm_clip_min_drop_fraction_of_center=fwhm_clip_min_drop_fraction_of_center,
                fwhm_clip_re_rise_fraction_of_center=fwhm_clip_re_rise_fraction_of_center,
                fwhm_branch_endpoint_exclusion_um=fwhm_branch_endpoint_exclusion_um,
                fwhm_junction_proximity_exclusion_um=fwhm_junction_proximity_exclusion_um,
                fwhm_enforce_same_edge_locality=fwhm_enforce_same_edge_locality,
                fwhm_same_edge_arc_window_um=fwhm_same_edge_arc_window_um,
                fwhm_same_edge_arc_window_multiplier=fwhm_same_edge_arc_window_multiplier,
                fwhm_same_edge_arc_window_min_um=fwhm_same_edge_arc_window_min_um,
                fwhm_cap_half_extent_by_nonlocal_same_edge_distance=fwhm_cap_half_extent_by_nonlocal_same_edge_distance,
                fwhm_nonlocal_same_edge_arc_separation_um=fwhm_nonlocal_same_edge_arc_separation_um,
                fwhm_nonlocal_same_edge_half_extent_factor=fwhm_nonlocal_same_edge_half_extent_factor,
                fwhm_reject_samples_with_center_offset=fwhm_reject_samples_with_center_offset,
                fwhm_max_fit_center_offset_um=fwhm_max_fit_center_offset_um,
                fwhm_reject_samples_with_low_fit_r2=fwhm_reject_samples_with_low_fit_r2,
                fwhm_min_fit_r2=fwhm_min_fit_r2,
            )
            G, haemo_results = apply_poiseuille_haemodynamics(G, config=haemo_config)
            if "fwhm" in haemo_results:
                print(f"FWHM diameter measurement summary: {haemo_results['fwhm']}")
                if do_pericyte_constriction:
                    print(
                        "Pericyte mode: passive diameter d1 from per-edge FWHM where available, "
                        "else DIAMETER_BY_BRANCH_ORDER; d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER."
                    )
            elif use_fwhm_edge_diameters is False:
                print(
                    "Vessel diameters: manual mode (DIAMETER_BY_BRANCH_ORDER / "
                    "set_poiseuille_resistances without per-edge FWHM)."
                )
            if "pericyte_comparison" in haemo_results:
                comparison_results = haemo_results["pericyte_comparison"]
                print(
                    "Pericyte resistance comparison complete: "
                    f"baseline={comparison_results['baseline_resistance']:.6f}, "
                    f"constricted={comparison_results['constricted_resistance']:.6f}, "
                    f"delta={comparison_results['delta']:.6f}, "
                    f"change={comparison_results['percent_change']:.3f}%."
                )
                print(
                    "Saved pericyte resistance comparison CSV to: "
                    f"{comparison_results['output_csv_path']}"
                )
            weight_results = haemo_results.get("weights", {})
            for step_name, step_result in weight_results.items():
                print(f"Haemodynamics weights [{step_name}]: {step_result}")

    # 5) Export vessels/pericytes/nodes to VTK and optionally visualize in PyVista.
    # FA I have no idea if pericyte location is correct. AI did that part.
    # FA I don't fully understand how pericyte location is currently determined?
    if run_haemodynamics and VTK_export:
        vtk_export = visualization.graph_to_vtk(G, vtk_output_prefix)
        print("\n=== VTK Export ===")
        print(f"  Vessels:   {vtk_export['vessels_path']}")
        print(f"  Pericytes: {vtk_export['pericytes_path']}")
        print(f"  Nodes:     {vtk_export['nodes_path']}")
        print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if run_haemodynamics and visualize_vtk and VTK_export:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )
    if run_haemodynamics and visualize_vtk and not VTK_export:
        print("VTK visualization requested but VTK export is disabled. Set VTK_export=True to enable.")
    if run_haemodynamics and not visualize_vtk:
        print("VTK visualization skipped.") 
    # 6) Compute effective resistance between two selected nodes.
    if run_haemodynamics:
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        print(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
    if run_haemodynamics and do_equiv_resistance_calculation:
        source_node, target_node = resistance_node_pair
        if source_node in node_to_idx and target_node in node_to_idx:
            laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
            two_point_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
                laplacian,
                G,
                source_node,
                target_node,
            )
            print(
                f"\nEffective resistance between nodes {source_node} and "
                f"{target_node}: {two_point_resistance}"
            )
        else:
            print(
                f"\nSkipped two-point resistance: nodes {resistance_node_pair} "
                "are not both present in the graph."
            )

    # 7) Compute and print vessel statistics.
    print("\nComputing vessel statistics...")
    if STATISTICS:
        valid_statistics_modes = {"fast", "full"}
        if statistics_mode not in valid_statistics_modes:
            raise ValueError(
                f"Invalid statistics_mode='{statistics_mode}'. "
                f"Choose one of {sorted(valid_statistics_modes)}."
            )
        node_positions = nx.get_node_attributes(G, "pos")
        stats = statistics.compute_comprehensive_vessel_statistics(
            G,
            node_positions=node_positions,
            image_dimensions=image.shape,
            statistics_mode=statistics_mode,
        )

        print("\n=== Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        stats_csv_path = output_dir / f"{image_path.stem}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        print(f"Saved statistics CSV to: {stats_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{image_path.stem}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        print(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")

        if run_haemodynamics:
            weighted_measurements = statistics.compute_betweenness_and_community_measurements(G)
        else:
            weighted_measurements = {
                "edge_resistance": {
                    "Betweenness": {
                        "Betweenness Mean": "N/A (haemodynamics disabled)",
                        "Betweenness Max": "N/A (haemodynamics disabled)",
                        "Betweenness Top Nodes": "N/A (haemodynamics disabled)",
                        "Betweenness Method": "N/A (haemodynamics disabled)",
                    },
                    "Communities": {
                        "Community Count": "N/A (haemodynamics disabled)",
                        "Largest Community Size": "N/A (haemodynamics disabled)",
                        "Mean Community Size": "N/A (haemodynamics disabled)",
                        "Community Method": "N/A (haemodynamics disabled)",
                    },
                },
                "edge_length": {
                    "Betweenness": statistics.compute_weighted_betweenness_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                    "Communities": statistics.compute_weighted_communities_summary(
                        G,
                        source_attr="length",
                        inverse_source_attr=False,
                    ),
                },
            }
        print("\n=== Weighted Betweenness and Communities ===")
        for model_name, model_results in weighted_measurements.items():
            print(f"  [{model_name}]")
            for metric_name, metric_values in model_results.items():
                print(f"    {metric_name}: {metric_values}")

        resistance_path = output_dir / f"{image_path.stem}_betweenness_communities_resistance.json"
        resistance_path.write_text(
            json.dumps(weighted_measurements["edge_resistance"], indent=2)
        )
        length_path = output_dir / f"{image_path.stem}_betweenness_communities_edge_length.json"
        length_path.write_text(
            json.dumps(weighted_measurements["edge_length"], indent=2)
        )
        print(f"Saved edge-resistance stats to: {resistance_path}")
        print(f"Saved edge-length stats to: {length_path}")
    else:
        print("Vessel statistics skipped.")

    # 8) Optional: nearest 3D distance from objects in a cell mask to vessel edge.
    if measurement_3d_to_cell_mask:
        if cell_mask_path is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(cell_mask_path),
            output_dir=output_dir,
            image_stem=image_path.stem,
            voxel_size_xyz=main_voxel_size_xyz,
            axis_order=axis_order,
            vessel_mask_path=(
                None
                if measurement_3d_vessel_mask_path is None
                else Path(measurement_3d_vessel_mask_path)
            ),
            vessel_reference_image_path=(
                None
                if measurement_3d_reference_image_path is None
                else Path(measurement_3d_reference_image_path)
            ),
            cell_mask_h5_dataset_name=cell_mask_h5_dataset_name,
            vessel_mask_h5_dataset_name=measurement_3d_vessel_mask_h5_dataset_name,
            vessel_reference_h5_dataset_name=measurement_3d_reference_h5_dataset_name,
        )
        print(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        print("3D cell-mask vessel-distance measurement skipped.")

    # 9) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    if run_haemodynamics:
        print("\nSolving flow through the network...")
        flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            input_p_bc,
            output_p_bc,
            starting_nodes,
            output_nodes,
            vtk_export,
        )
        print("Flow through the network solved")
        print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")
    else:
        print("Haemodynamics solve skipped (run_haemodynamics=False).")

    # 10) Optional matplotlib visualization.
    if visualize_results:
        print("\nGenerating visualizations...")
        valid_plot_modes = {"all", "final_only", "none"}
        if ide_plot_mode not in valid_plot_modes:
            raise ValueError(
                f"Invalid ide_plot_mode='{ide_plot_mode}'. "
                f"Choose one of {sorted(valid_plot_modes)}."
            )
        show_any_ide_plot = show_plots_in_ide and ide_plot_mode != "none"
        show_degree_plot = show_plots_in_ide and ide_plot_mode == "all"
        show_overlay_plot = show_any_ide_plot and final_render_mode == "2d"
        show_3d_plot = show_any_ide_plot and final_render_mode == "3d"
        show_branch_order_plot = show_plots_in_ide and ide_plot_mode == "all"
        visualization.plot_node_degree_distribution(
            G,
            save_path=None if interactive_plots else plot_dir / "node_degree_distribution.png",
            show=interactive_plots or show_degree_plot,
            show_after_save=show_degree_plot and not interactive_plots,
        )
        if final_render_mode == "3d":
            overlay_3d_path = None if interactive_plots else plot_dir / "edges_and_nodes_overlay_3d.html"
            visualization.visualize_3d_plotly(
                G,
                title="Edges and Nodes Overlay (Interactive 3D)",
                save_html_path=str(overlay_3d_path) if overlay_3d_path else None,
                show=interactive_plots or show_3d_plot,
            )
            if overlay_3d_path is not None:
                print(f"Saved interactive 3D overlay to: {overlay_3d_path}")
        else:
            visualization.visualize_edges_and_nodes(
                image,
                G,
                save_path=None if interactive_plots else plot_dir / "edges_and_nodes_overlay.png",
                show=interactive_plots or show_overlay_plot,
                show_after_save=show_overlay_plot and not interactive_plots,
            )
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if starting_nodes:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
                save_path=None if interactive_plots else plot_dir / "geometry_with_branch_orders.png",
                show=interactive_plots or show_branch_order_plot,
                show_after_save=show_branch_order_plot and not interactive_plots,
            )
        if (
            hold_ide_plots_open
            and show_any_ide_plot
            and not interactive_plots
            and plt.get_fignums()
        ):
            print("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        print("Matplotlib visualizations skipped.")


# ---------------------------------------------------------------------------
# Settings -> pipeline arguments
#
# `resistance_pipeline_config.yaml` is the source of every setting, described by
# `resistance_pipeline_schema.py`. This section is the only place that knows how
# a setting name maps onto the pipeline stage arguments.
# ---------------------------------------------------------------------------

CONFIG_PATH = examples_dir / "resistance_pipeline_config.yaml"

#: Arguments the stages take, captured once so the mapping does not depend on
#: re-introspecting a function that may have been wrapped or patched.
STAGE_PARAMETERS = frozenset(inspect.signature(run_pipeline_stages).parameters)

#: Settings whose name differs from the stage argument they feed. Before the
#: schema these were an ad-hoc alias table that `axis_order` was missing from,
#: so `IMAGE_AXIS_ORDER` from a config file was silently ignored.
SETTING_TO_ARGUMENT = {
    "input_path": "image_path",
    "image_axis_order": "axis_order",
    "do_pericyte_construction": "do_pericyte_constriction",
}

#: Settings the stage body reads from module globals rather than its arguments.
SETTING_TO_GLOBAL = {
    "vtk_export": "VTK_export",
    "statistics": "STATISTICS",
    "custom_edges": "custom_edges",
}

#: Settings consumed here rather than passed on: they build the derived tables
#: and the plot directory.
_DERIVED_INPUTS = frozenset(
    {
        "all_diams_const",
        "max_branch_order",
        "default_diameter",
        "manual_capillary_diameter_by_branch_order",
        "manual_arteriole_diameter_by_branch_order",
        "manual_venule_diameter_by_branch_order",
        "base_plot_dir",
        "use_volume_boxes",
    }
)


def resolve_settings(
    settings: dict | None = None,
    *,
    overrides: dict | None = None,
    config_path: Path | str = CONFIG_PATH,
) -> dict:
    """Every setting for one run, validated against the schema.

    With no arguments this is exactly what the config file says. Pass
    ``settings`` to supply an already-loaded dict, and ``overrides`` to change
    individual values on top of either.
    """
    if settings is None:
        resolved = load_config(config_path, SCHEMA, overrides=overrides)
    else:
        merged = {**settings, **(overrides or {})}
        resolved = SCHEMA.validate(merged)
    _fill_derived_settings(resolved)
    return resolved


def _fill_derived_settings(settings: dict) -> None:
    """Build the branch-order tables when the config leaves them unset.

    They are functions of the manual diameter settings, so the config file
    states those and leaves these null rather than duplicating a 150-entry
    table that could then disagree with them.
    """
    if settings.get("diameter_by_branch_order") is None:
        settings["diameter_by_branch_order"] = haemodynamics.build_diameter_by_branch_order(
            all_diams_const=settings["all_diams_const"],
            max_branch_order=settings["max_branch_order"],
            default_diameter=settings["default_diameter"],
            manual_capillary_diameter_by_branch_order=settings[
                "manual_capillary_diameter_by_branch_order"
            ],
            manual_arteriole_diameter_by_branch_order=settings[
                "manual_arteriole_diameter_by_branch_order"
            ],
            manual_venule_diameter_by_branch_order=settings[
                "manual_venule_diameter_by_branch_order"
            ],
        )
    if settings.get("constriction_by_branch_order") is None:
        max_order = int(settings["max_branch_order"])
        constriction = {"B01": 1.0, "Art1": 1.0, "Ven1": 1.0}
        for order in range(2, max_order + 1):
            constriction[f"B{order:02d}"] = 0.8
            constriction[f"Art{order}"] = 0.8
            constriction[f"Ven{order}"] = 0.8
        settings["constriction_by_branch_order"] = constriction


def stage_arguments(settings: dict) -> dict:
    """Translate a settings dict into the arguments the stages take."""
    arguments: dict[str, object] = {}
    for name, value in settings.items():
        if name in _DERIVED_INPUTS or name in SETTING_TO_GLOBAL:
            continue
        argument = SETTING_TO_ARGUMENT.get(name, name)
        if argument in STAGE_PARAMETERS:
            arguments[argument] = value
    arguments["plot_dir"] = Path(settings["base_plot_dir"]) / "nerve"
    return arguments


#: Stage arguments that are not settings, so may be overridden but not configured.
_STAGE_ONLY_ARGUMENTS = ("plot_dir",)

#: Reverse of SETTING_TO_ARGUMENT, so a caller may use either spelling.
ARGUMENT_TO_SETTING = {
    argument: setting for setting, argument in SETTING_TO_ARGUMENT.items()
}


def _split_overrides(overrides: dict) -> tuple[dict, dict]:
    """Separate setting overrides from stage-only ones, accepting either name."""
    setting_overrides: dict[str, object] = {}
    stage_overrides: dict[str, object] = {}
    for key, value in overrides.items():
        if key in _STAGE_ONLY_ARGUMENTS:
            stage_overrides[key] = value
        else:
            setting_overrides[ARGUMENT_TO_SETTING.get(key, key)] = value
    return setting_overrides, stage_overrides


def image_to_model_pipeline(settings: dict | None = None, **overrides):
    """Run the pipeline for one settings dict.

    The pipeline reads far more than six settings, so the project convention is
    to hand it the whole dict; ``overrides`` changes individual values for a
    single call without editing the config file::

        image_to_model_pipeline()                                # the config file
        image_to_model_pipeline(settings)                        # a loaded dict
        image_to_model_pipeline(image_path=..., do_skeletonize=False)   # one-offs
    """
    setting_overrides, stage_overrides = _split_overrides(overrides)
    resolved = resolve_settings(settings, overrides=setting_overrides or None)

    # Three settings are read from module globals inside the stages rather than
    # passed as arguments, so a config change to them would otherwise be lost.
    for setting_name, global_name in SETTING_TO_GLOBAL.items():
        globals()[global_name] = resolved[setting_name]

    arguments = stage_arguments(resolved)
    arguments.update(stage_overrides)
    return run_pipeline_stages(**arguments)


def _build_pipeline_kwargs_from_active_settings(plot_dir: Path) -> dict:
    """Build full pipeline kwargs from current module-level settings."""
    alias_to_settings = {
        "image_path": "INPUT_PATH",
        "do_pericyte_constriction": "DO_PERICYTE_CONSTRUCTION",
    }
    kwargs: dict = {}
    for param_name in inspect.signature(run_pipeline_stages).parameters:
        if param_name == "plot_dir":
            kwargs[param_name] = plot_dir
            continue
        setting_name = alias_to_settings.get(param_name, param_name.upper())
        if setting_name in globals():
            kwargs[param_name] = globals()[setting_name]
    return kwargs


def _parse_cli_literal(value_text: str) -> object:
    lowered = value_text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value_text)
    except (ValueError, SyntaxError):
        return value_text


def _coerce_pipeline_cli_value(param_name: str, value_text: str) -> object:
    value = _parse_cli_literal(value_text)
    if isinstance(value, str):
        if (
            param_name.endswith("_path")
            or param_name.endswith("_dir")
            or param_name.endswith("_prefix")
            or param_name == "plot_dir"
        ):
            return Path(value)
    return value


def _extract_pipeline_cli_overrides(cli_namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for param_name in inspect.signature(run_pipeline_stages).parameters:
        dest = f"pipeline_arg__{param_name}"
        if not hasattr(cli_namespace, dest):
            continue
        raw_value = getattr(cli_namespace, dest)
        if raw_value is None:
            continue
        overrides[param_name] = _coerce_pipeline_cli_value(param_name, raw_value)
    return overrides


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the resistance network pipeline. Settings come from "
            "resistance_pipeline_config.yaml; every one of them can be overridden "
            "with a flag of the same name for a single run."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="YAML config to run from (default: %(default)s).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESET_DEFINITIONS.keys()),  # noqa: F405
        help="Apply a named preset's overrides on top of the config file.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    parser.add_argument(
        "--list-settings",
        action="store_true",
        help="List every setting with its value for this run, and exit.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Write the settings this run would use to a YAML file, and exit.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the preflight checklist and exit without executing the pipeline.",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Answer setup prompts instead of editing the config file.",
    )
    # One flag per setting, generated from the schema.
    add_schema_arguments(parser, SCHEMA)
    cli = parser.parse_args()

    if cli.list_presets:
        print("Available presets:")
        for preset_name, description in list_presets().items():  # noqa: F405
            print(f"  - {preset_name}: {description}")
        raise SystemExit(0)

    overrides: dict[str, object] = {}
    if cli.preset:
        # Preset overrides are still written in SCREAMING_SNAKE; the schema
        # names are the lower-case form of the same settings.
        preset_settings = build_settings_for_preset(preset_name=cli.preset)  # noqa: F405
        overrides.update(
            {
                name.lower(): value
                for name, value in preset_settings.items()
                if name.lower() in SCHEMA
            }
        )
        print(f"Applying preset '{cli.preset}'")
    if cli.wizard:
        wizard_results = run_interactive_setup_wizard(
            default_preset=cli.preset or "default",
            available_presets=sorted(PRESET_DEFINITIONS.keys()),  # noqa: F405
        )
        overrides.update(
            {
                name.lower(): value
                for name, value in wizard_results["settings_overrides"].items()
                if name.lower() in SCHEMA
            }
        )
        for name, value in wizard_results["pipeline_overrides"].items():
            setting_name = ARGUMENT_TO_SETTING.get(name, name)
            if setting_name in SCHEMA:
                overrides[setting_name] = value
    overrides.update(cli_overrides(cli))

    settings = resolve_settings(config_path=cli.config, overrides=overrides or None)
    print(f"Settings from: {cli.config}")
    if overrides:
        print(f"Overridden for this run: {sorted(overrides)}")

    if cli.list_settings:
        for section, section_settings in SCHEMA.sections().items():
            print(f"\n{section}")
            for setting in section_settings:
                print(f"  {setting.name:52s} {settings[setting.name]!r}")
        raise SystemExit(0)

    if cli.save_config is not None:
        saved_path = dump_config(cli.save_config, SCHEMA, values=settings)
        print(f"Saved the settings for this run to: {saved_path}")
        raise SystemExit(0)

    preflight_report = run_preflight_checklist(stage_arguments(settings))
    if not preflight_report["ok"]:
        raise SystemExit(2)
    if cli.preflight_only:
        print("Preflight-only mode: exiting before pipeline execution.")
        raise SystemExit(0)

    image_to_model_pipeline(settings)
