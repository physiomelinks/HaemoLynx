"""The image-to-model pipeline, stage by stage.

Segmentation, skeletonisation, graph building, boundary and branch-order
assignment, haemodynamics, export and statistics — driven by one settings dict
as loaded from a config file.

Lives here rather than in an example so that more than one example can run it:
the whole-brain script adds a pericyte dilation sweep on top of exactly this.
"""
from __future__ import annotations

import inspect
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tifffile

from ImageLynx import graph, haemodynamics, io, preprocessing, statistics, visualization
from ImageLynx.haemodynamics.pipeline import (
    HaemodynamicsApplyConfig,
    apply_poiseuille_haemodynamics,
)
from ImageLynx.io.voxel_validation import resolve_voxel_size_xyz
from ImageLynx.parsers import Schema, parameters_of, prefixed_arguments

logger = logging.getLogger(__name__)


def run_pipeline_stages(settings: dict, schema: Schema) -> nx.MultiGraph | None:
    """Run every pipeline stage for one resolved settings dict.

    Returns the graph the run produced, so a caller can do more with it — the
    whole-brain script sweeps pericyte dilation over exactly this graph.

    Reads settings by name rather than taking them as arguments: there are 137
    of them, far past the point where a signature documents anything. See
    ``resistance_pipeline_schema.py`` for what each one means and
    ``resistance_pipeline_config.yaml`` for the values.
    """
    settings["input_path"] = Path(settings["input_path"])
    if settings["use_ilastik_segmentation"]:
        unsegmented_image_path = Path(settings["ilastik_unsegmented_image_path"])
        unsegmented_image_path = io.resolve_image_path_with_optional_zip(unsegmented_image_path)
        if settings["ilastik_classifier_path"] is None:
            raise ValueError(
                "ilastik_classifier_path must be set when use_ilastik_segmentation=True."
            )
        settings["ilastik_output_dir"] = Path(settings["ilastik_output_dir"])
        ilastik_segmented_path = settings["ilastik_output_dir"] / (
            f"{unsegmented_image_path.stem}_segmented{settings['ilastik_output_suffix']}"
        )
        print(f"Running ilastik segmentation for unsegmented image: {unsegmented_image_path}")
        settings["input_path"] = io.run_ilastik_headless_segmentation(
            input_image_path=unsegmented_image_path,
            classifier_path=Path(settings["ilastik_classifier_path"]),
            output_path=ilastik_segmented_path,
            ilastik_executable=settings["ilastik_executable"],
        )
        print(f"Using ilastik-segmented image: {settings['input_path']}")
    else:
        print(f"Using segmented input image: {settings['input_path']}")

    settings["input_path"] = io.resolve_image_path_with_optional_zip(settings["input_path"])
    # get image format from image_path
    input_format = settings["input_path"].suffix[1:].lower()
    if input_format not in ["tif", "tiff", "h5"]:
        raise ValueError(f"Invalid image format: {input_format}")
    settings["image_axis_order"] = io.normalize_axis_order(settings["image_axis_order"], label="IMAGE_AXIS_ORDER")
    if settings["image_axis_order"] != io.CANONICAL_AXIS_ORDER:
        print(
            f"Input axis order '{settings['image_axis_order']}' will be transposed to the canonical "
            f"'{io.CANONICAL_AXIS_ORDER}' layout on load."
        )
    settings["vtk_output_prefix"] = Path(settings["vtk_output_prefix"])
    output_dir = settings["vtk_output_prefix"].parent
    valid_final_render_modes = {"2d", "3d"}
    if settings["final_render_mode"] not in valid_final_render_modes:
        raise ValueError(
            f"Invalid final_render_mode='{settings['final_render_mode']}'. "
            f"Choose one of {sorted(valid_final_render_modes)}."
        )

    logging.basicConfig(
        level=logging.DEBUG if settings["verbose_logging"] else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # 1) Load image and skeletonize.
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / f"{settings['input_path'].stem}_skeleton.npy"
    voxel_meta_path = output_dir / f"{settings['input_path'].stem}_voxel_size.json"
    graph_path = output_dir / f"{settings['input_path'].stem}_graph.pkl"
    projection_path = settings["plot_dir"] / "skeleton_projection.png"
    if not settings["plot_dir"].exists():
        settings["plot_dir"].mkdir(parents=True, exist_ok=True)

    if settings["do_skeletonize"]:
        if input_format in {"tif", "tiff"}:
            (
                image,
                skeleton,
                voxel_size_x,
                voxel_size_y,
                voxel_size_z,
                voxel_meta_status,
            ) = io.load_and_skeletonize_3d_tif(
                settings["input_path"],
                axis_order=settings["image_axis_order"],
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
                settings["input_path"],
                axis_order=settings["image_axis_order"],
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
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
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
            component_connectivity=settings["skeleton_component_connectivity"],
        )
        visualization.visualize_skeleton(skeleton, save_path=settings["plot_dir"] / "raw_skeleton.png")

        # The `skeleton_*` settings are this function's parameters with a
        # prefix, so they go in as a group; the percentage is the one exception.
        skeleton = preprocessing.preprocess_skeleton_for_graph(
            skeleton,
            **prefixed_arguments(
                settings,
                "skeleton_",
                parameters_of(preprocessing.preprocess_skeleton_for_graph),
            ),
            min_component_fraction=settings["skeleton_min_component_percent"] / 100.0,
        )
        preprocessing.print_skeleton_connectivity_stats(
            "cleaned",
            skeleton,
            component_connectivity=settings["skeleton_component_connectivity"],
        )
        
        # save the skeleton
        np.save(skeleton_path, skeleton)
        voxel_meta_path.write_text(
            json.dumps(
                {
                    "voxel_size": voxel_size,
                    "voxel_size_source": voxel_size_source,
                    "voxel_metadata_status": voxel_meta_status,
                    "voxel_size_policy": settings["voxel_size_policy"],
                    "voxel_size_override_xyz": settings["voxel_size_override_xyz"],
                }
            )
        )
        print(f"Saved skeleton to: {skeleton_path}")
    else:
        # load the skeleton
        skeleton = np.load(skeleton_path)
        image = io.apply_axis_order(tifffile.imread(settings["input_path"]), settings["image_axis_order"])
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
            voxel_size_override_xyz=settings["voxel_size_override_xyz"],
            voxel_size_policy=settings["voxel_size_policy"],
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
    # The vessel-mask and segmentation settings go in as a group; each role
    # picks the ones it uses. `io.VESSEL_MASK_SETTINGS` lists which those are.
    mask_settings = {
        **schema.section_values(settings, "Vessel masks"),
        **schema.section_values(settings, "Input and segmentation"),
    }
    (
        large_arteriole_mask,
        large_venule_mask,
        large_arteriole_mask_voxel_size,
        large_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        **io.vessel_mask_arguments(mask_settings, "large"),
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
    )
    (
        small_arteriole_mask,
        small_venule_mask,
        small_arteriole_mask_voxel_size,
        small_venule_mask_voxel_size,
    ) = io.load_and_validate_vessel_masks(
        **io.vessel_mask_arguments(mask_settings, "small"),
        image_shape=image.shape,
        main_voxel_size_xyz=main_voxel_size_xyz,
        loaded_message_suffix=(
            f"min_overlap_fraction={float(settings['small_vessel_mask_min_overlap_fraction']):.3f}"
        ),
    )

    if settings["do_graph_building"]:
        # 3) Convert skeleton to graph.
        def _graph_build_step_callback(graph_obj, label: str) -> None:
            plot_png = label
            if label == "smart_multigraph_degree2_removal_pass1":
                plot_png = "smart_multigraph_degree2_removal"
            visualization.save_graph_snapshot(
                graph_obj,
                image,
                output_dir,
                settings["plot_dir"],
                settings["input_path"].stem,
                label,
            )
            visualization.visualize_edges_and_nodes(
                image,
                graph_obj,
                label_nodes=True,
                save_path=settings["plot_dir"] / f"{plot_png}.png",
            )

        G = graph.build_graph_from_skeleton(
            skeleton,
            voxel_size=voxel_size_zyx,
            graph_reconnect_threshold=settings["graph_reconnect_threshold"],
            final_orphan_reconnect_threshold=settings["final_orphan_reconnect_threshold"],
            cluster_collapse_distance=settings["cluster_collapse_distance"],
            min_stub_length=settings["min_stub_length"],
            debug=settings["verbose_logging"],
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
    if settings["final_render_mode"] == "3d":
        final_graph_3d_path = settings["plot_dir"] / "final_graph_3d.html"
        visualization.visualize_3d_plotly(
            G,
            title="Final Graph (Interactive 3D)",
            save_html_path=str(final_graph_3d_path),
            show=settings["show_plots_in_ide"] or settings["interactive_plots"],
        )
        print(f"Saved interactive 3D final graph to: {final_graph_3d_path}")
    else:
        visualization.visualize_edges_and_nodes(
            image,
            G,
            label_nodes=False,
            save_path=settings["plot_dir"] / "final_graph.png",
            show_coordinates_degree_1=True,
        )

    auto_start_nodes: list[int] = []
    auto_output_nodes: list[int] = []
    if settings["automated_vessel_assignment"]:
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
        settings["starting_node_coordinates"] = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_start_nodes
        ]
        settings["output_node_coordinates"] = [
            tuple(np.asarray(G.nodes[node_id]["pos"], dtype=float))
            for node_id in auto_output_nodes
        ]
        automated_assignment_html_path = settings["plot_dir"] / "automated_vessel_assignment_3d.html"
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
            f"{len(settings['starting_node_coordinates'])} input coordinates from arteriole-mask overlap "
            f"and {len(settings['output_node_coordinates'])} output coordinates from venule-mask overlap."
        )

    settings["starting_nodes"][:] = []
    settings["output_nodes"][:] = []
    settings["arteriole_boundary_nodes"][:] = []
    settings["venule_boundary_nodes"][:] = []
    if settings["automated_vessel_assignment"]:
        # Use direct terminal-node overlap assignment from vessel masks.
        start_nodes = auto_start_nodes
        out_nodes = [node_id for node_id in auto_output_nodes if node_id not in set(start_nodes)]
    else:
        # Each role reads its own three settings; naming the role is enough.
        start_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "starting"
        )
        out_nodes = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "output", exclude_nodes=start_nodes
        )
    settings["starting_nodes"].extend(start_nodes)
    settings["output_nodes"].extend(out_nodes)
    used_nodes = set(settings["starting_nodes"]) | set(settings["output_nodes"])
    if settings["arteriole_boundary_node_coordinates"] or settings["arteriole_boundary_node_volumes"]:
        art_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "arteriole_boundary", exclude_nodes=list(used_nodes)
        )
        settings["arteriole_boundary_nodes"].extend(art_boundary)
        used_nodes.update(settings["arteriole_boundary_nodes"])

    if settings["venule_boundary_node_coordinates"] or settings["venule_boundary_node_volumes"]:
        ven_boundary = graph.select_boundary_nodes_for_role(
            G, image.shape, settings, "venule_boundary", exclude_nodes=list(used_nodes)
        )
        settings["venule_boundary_nodes"].extend(ven_boundary)
    if settings["use_small_vessel_masks_for_boundary_assignment"]:
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
            minimum_overlap_fraction=float(settings["small_vessel_mask_min_overlap_fraction"]),
            allow_overlap=False,
        )
        settings["arteriole_boundary_nodes"][:] = list(
            inferred_boundary_results["arteriole_boundary_nodes"]
        )
        settings["venule_boundary_nodes"][:] = list(inferred_boundary_results["venule_boundary_nodes"])
        print(
            "Small-vessel mask boundary assignment selected "
            f"{len(settings['arteriole_boundary_nodes'])} arteriole boundary nodes and "
            f"{len(settings['venule_boundary_nodes'])} venule boundary nodes "
            f"(min_overlap_fraction={float(settings['small_vessel_mask_min_overlap_fraction']):.3f})."
        )
        print(
            "Small-vessel mask edge labels: "
            f"arteriole_edges={inferred_boundary_results['arteriole_edge_count']}, "
            f"venule_edges={inferred_boundary_results['venule_edge_count']}, "
            f"overlap_edges={inferred_boundary_results['overlap_edge_count']}."
        )
        if settings["write_small_vessel_boundary_labelling_3d_html"]:
            boundary_html = Path(settings["plot_dir"]) / "small_vessel_mask_boundary_labelling_3d.html"
            Path(settings["plot_dir"]).mkdir(parents=True, exist_ok=True)
            ok = graph.write_small_vessel_mask_boundary_labelling_3d_html(
                G,
                small_arteriole_mask=small_arteriole_mask,
                small_venule_mask=small_venule_mask,
                arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
                venule_boundary_nodes=settings["venule_boundary_nodes"],
                voxel_size_zyx=voxel_size_zyx,
                output_html_path=boundary_html,
            )
            if ok:
                print(f"Saved interactive 3D small-vessel boundary view: {boundary_html}")
            else:
                print(
                    "Small-vessel boundary 3D HTML not written (install plotly to enable)."
                )
    if settings["automated_vessel_assignment"]:
        print(
            f"Selected {len(settings['starting_nodes'])} STARTING_NODES and {len(settings['output_nodes'])} "
            "OUTPUT_NODES directly from terminal-node overlap with vessel masks."
        )
    else:
        print(
            f"Selected {len(settings['starting_nodes'])} STARTING_NODES and {len(settings['output_nodes'])} "
            "OUTPUT_NODES from manual coordinates."
        )
    print(f"Starting nodes are: {settings['starting_nodes']}")
    print(f"Output nodes are: {settings['output_nodes']}")
    print(f"Arteriole boundary nodes are: {settings['arteriole_boundary_nodes']}")
    print(f"Venule boundary nodes are: {settings['venule_boundary_nodes']}")

    if settings["starting_nodes"] and settings["output_nodes"]:
        resistance_node_pair = (settings["starting_nodes"][0], settings["output_nodes"][0])
        print(f"Auto-selected resistance node pair: {resistance_node_pair}")
    else:
        if settings["automated_vessel_assignment"]:
            raise ValueError(
                "No starting or output nodes found from terminal-node overlap with "
                "arteriole/venule masks."
            )
        raise ValueError(
            "No starting or output nodes found from manual input coordinates."
        )

    # 4) Add branch orders and hemodynamic edge weights.
    if settings["starting_nodes"]:

        def _vessel_types_after_branch_assign(graph_obj) -> None:
            vessel_type_3d_path = settings["plot_dir"] / "vessel_types_assigned_3d.html"
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
            settings["starting_nodes"],
            output_nodes=settings["output_nodes"],
            arteriole_boundary_nodes=settings["arteriole_boundary_nodes"],
            venule_boundary_nodes=settings["venule_boundary_nodes"],
            strict_hierarchical=settings["strict_branch_order_assignment"],
            expects_hierarchical=bool(
                settings["automated_vessel_assignment"]
                or settings["use_small_vessel_masks_for_boundary_assignment"]
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

        if not settings["run_haemodynamics"]:
            print(
                "Haemodynamics disabled; skipping diameter fitting and "
                "Poiseuille conductance assignment."
            )
        elif settings["run_haemodynamics"]:
            # Two config sections go in whole rather than as forty-odd
            # keyword arguments; everything else here is computed by this run.
            haemo_config = HaemodynamicsApplyConfig(
                diameters=schema.section_values(settings, "Diameters and pericytes"),
                fwhm=schema.section_values(settings, "FWHM diameter measurement"),
                resistance_node_pair=resistance_node_pair,
                voxel_size_zyx=voxel_size_zyx,
                axis_order=settings["image_axis_order"],
                comparison_output_csv_path=(
                    output_dir / f"{settings['input_path'].stem}_pericyte_resistance_comparison.csv"
                    if settings["run_pericyte_resistance_comparison"]
                    else None
                ),
            )
            G, haemo_results = apply_poiseuille_haemodynamics(G, config=haemo_config)
            if "fwhm" in haemo_results:
                print(f"FWHM diameter measurement summary: {haemo_results['fwhm']}")
                if settings["do_pericyte_construction"]:
                    print(
                        "Pericyte mode: passive diameter d1 from per-edge FWHM where available, "
                        "else DIAMETER_BY_BRANCH_ORDER; d2 = d1 * CONSTRICTION_BY_BRANCH_ORDER."
                    )
            elif settings["use_fwhm_edge_diameters"] is False:
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
    if settings["run_haemodynamics"] and settings["vtk_export"]:
        vtk_export = visualization.graph_to_vtk(G, settings["vtk_output_prefix"])
        print("\n=== VTK Export ===")
        print(f"  Vessels:   {vtk_export['vessels_path']}")
        print(f"  Pericytes: {vtk_export['pericytes_path']}")
        print(f"  Nodes:     {vtk_export['nodes_path']}")
        print(f"  Counts: vessels={vtk_export['vessel_line_count']}, "
          f"pericytes={vtk_export['pericyte_count']}, nodes={vtk_export['node_count']}")
    if settings["run_haemodynamics"] and settings["visualize_vtk"] and settings["vtk_export"]:
        visualization.visualize_vtk_network(
            vtk_export["vessels_path"],
            vtk_export["pericytes_path"],
            vtk_export["nodes_path"],
            show_nodes=False,
        )
    if settings["run_haemodynamics"] and settings["visualize_vtk"] and not settings["vtk_export"]:
        print(
            "VTK visualization requested but VTK export is disabled. "
            "Set vtk_export: true in the config to enable."
        )
    if settings["run_haemodynamics"] and not settings["visualize_vtk"]:
        print("VTK visualization skipped.") 
    # 6) Compute effective resistance between two selected nodes.
    if settings["run_haemodynamics"]:
        conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
        print(f"Conductance matrix built with shape {conductance.shape} and node_list length {len(node_list)}.")
    if settings["run_haemodynamics"] and settings["do_equiv_resistance_calculation"]:
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
    if settings["statistics"]:
        valid_statistics_modes = {"fast", "full"}
        if settings["statistics_mode"] not in valid_statistics_modes:
            raise ValueError(
                f"Invalid statistics_mode='{settings['statistics_mode']}'. "
                f"Choose one of {sorted(valid_statistics_modes)}."
            )
        node_positions = nx.get_node_attributes(G, "pos")
        stats = statistics.compute_comprehensive_vessel_statistics(
            G,
            node_positions=node_positions,
            image_dimensions=image.shape,
            statistics_mode=settings["statistics_mode"],
        )

        print("\n=== Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        stats_csv_path = output_dir / f"{settings['input_path'].stem}_statistics.csv"
        statistics.export_statistics_to_csv(stats, stats_csv_path)
        print(f"Saved statistics CSV to: {stats_csv_path}")

        branch_stats = statistics.compute_branch_order_statistics(
            G,
            node_positions=node_positions,
        )
        branch_stats_csv_path = output_dir / f"{settings['input_path'].stem}_branch_statistics.csv"
        statistics.export_branch_order_statistics_to_csv(
            branch_stats,
            branch_stats_csv_path,
        )
        print(f"Saved branch-order statistics CSV to: {branch_stats_csv_path}")

        if settings["run_haemodynamics"]:
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

        resistance_path = output_dir / f"{settings['input_path'].stem}_betweenness_communities_resistance.json"
        resistance_path.write_text(
            json.dumps(weighted_measurements["edge_resistance"], indent=2)
        )
        length_path = output_dir / f"{settings['input_path'].stem}_betweenness_communities_edge_length.json"
        length_path.write_text(
            json.dumps(weighted_measurements["edge_length"], indent=2)
        )
        print(f"Saved edge-resistance stats to: {resistance_path}")
        print(f"Saved edge-length stats to: {length_path}")
    else:
        print("Vessel statistics skipped.")

    # 8) Optional: nearest 3D distance from objects in a cell mask to vessel edge.
    if settings["measurement_3d_to_cell_mask"]:
        if settings["cell_mask_path"] is None:
            raise ValueError(
                "measurement_3d_to_cell_mask=True requires cell_mask_path."
            )
        distance_summary = statistics.run_3d_measurement_to_cell_mask(
            graph=G,
            cell_mask_path=Path(settings["cell_mask_path"]),
            output_dir=output_dir,
            image_stem=settings["input_path"].stem,
            voxel_size_xyz=main_voxel_size_xyz,
            axis_order=settings["image_axis_order"],
            vessel_mask_path=(
                None
                if settings["measurement_3d_vessel_mask_path"] is None
                else Path(settings["measurement_3d_vessel_mask_path"])
            ),
            vessel_reference_image_path=(
                None
                if settings["measurement_3d_reference_image_path"] is None
                else Path(settings["measurement_3d_reference_image_path"])
            ),
            cell_mask_h5_dataset_name=settings["cell_mask_h5_dataset_name"],
            vessel_mask_h5_dataset_name=settings["measurement_3d_vessel_mask_h5_dataset_name"],
            vessel_reference_h5_dataset_name=settings["measurement_3d_reference_h5_dataset_name"],
        )
        print(
            "3D cell-mask vessel-distance summary: "
            f"{distance_summary}"
        )
    else:
        print("3D cell-mask vessel-distance measurement skipped.")

    # 9) Also solve for flow throughout the network using the conductance matrix 
    # and the input and output pressures.
    if settings["run_haemodynamics"]:
        print("\nSolving flow through the network...")
        flow, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
            conductance,
            node_list,
            settings["input_p_bc"],
            settings["output_p_bc"],
            settings["starting_nodes"],
            settings["output_nodes"],
            vtk_export,
        )
        print("Flow through the network solved")
        print(f"Vtk file with flow data saved to: {vtk_export['vessels_path']}")
    else:
        print("Haemodynamics solve skipped (run_haemodynamics=False).")

    # 10) Optional matplotlib visualization.
    if settings["visualize_results"]:
        print("\nGenerating visualizations...")
        valid_plot_modes = {"all", "final_only", "none"}
        if settings["ide_plot_mode"] not in valid_plot_modes:
            raise ValueError(
                f"Invalid ide_plot_mode='{settings['ide_plot_mode']}'. "
                f"Choose one of {sorted(valid_plot_modes)}."
            )
        show_any_ide_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] != "none"
        show_degree_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] == "all"
        show_overlay_plot = show_any_ide_plot and settings["final_render_mode"] == "2d"
        show_3d_plot = show_any_ide_plot and settings["final_render_mode"] == "3d"
        show_branch_order_plot = settings["show_plots_in_ide"] and settings["ide_plot_mode"] == "all"
        visualization.plot_node_degree_distribution(
            G,
            save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "node_degree_distribution.png",
            show=settings["interactive_plots"] or show_degree_plot,
            show_after_save=show_degree_plot and not settings["interactive_plots"],
        )
        if settings["final_render_mode"] == "3d":
            overlay_3d_path = None if settings["interactive_plots"] else settings["plot_dir"] / "edges_and_nodes_overlay_3d.html"
            visualization.visualize_3d_plotly(
                G,
                title="Edges and Nodes Overlay (Interactive 3D)",
                save_html_path=str(overlay_3d_path) if overlay_3d_path else None,
                show=settings["interactive_plots"] or show_3d_plot,
            )
            if overlay_3d_path is not None:
                print(f"Saved interactive 3D overlay to: {overlay_3d_path}")
        else:
            visualization.visualize_edges_and_nodes(
                image,
                G,
                save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "edges_and_nodes_overlay.png",
                show=settings["interactive_plots"] or show_overlay_plot,
                show_after_save=show_overlay_plot and not settings["interactive_plots"],
            )
        #HD note - need visualisation of pericyte localisations (ie based upon constriction data)
        
        if settings["starting_nodes"]:
            visualization.visualize_geometry_with_branch_orders(
                image,
                G,
                group_above=8,
                save_path=None if settings["interactive_plots"] else settings["plot_dir"] / "geometry_with_branch_orders.png",
                show=settings["interactive_plots"] or show_branch_order_plot,
                show_after_save=show_branch_order_plot and not settings["interactive_plots"],
            )
        if (
            settings["hold_ide_plots_open"]
            and show_any_ide_plot
            and not settings["interactive_plots"]
            and plt.get_fignums()
        ):
            print("Holding plot windows open. Close them to finish the script.")
            plt.show(block=True)
    else:
        print("Matplotlib visualizations skipped.")

    return G
