"""Hover text for the panel's setting rows, written for someone new to it.

A setting's ``help`` is its config-file comment and command-line help and
says everything about it. The panel shows this instead: what the row does
and which way to set it, in a sentence or two, with no setting names, code
or history (the voice ``tests/test_gui_tooltips.py`` holds it to). Every
setting of :func:`haemolynx.pipeline.default_schema` has an entry; a setting
with none, such as one an example's schema adds, shows its ``help``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haemolynx.parsers.schema import Setting

SETTING_TOOLTIPS: dict[str, str] = {
    # 1. Input
    "network_source": (
        "What to analyse: a segmented image (default), a synthetic network matched to an earlier"
        " run, or that run's own network with a synthetic one drawn round it"
    ),
    "synthetic_template_path": (
        "The finished image run, saved with Statistics on, that the synthetic network is matched"
        " to; or a batch run's .haemobatch file, averaging its condition's runs"
    ),
    "synthetic_template_source": (
        "Where the template's characteristics are read from: its saved statistics report "
        "(default), or per-vessel distributions measured off its network"
    ),
    "synthetic_seed": (
        "Random seed for drawing the synthetic network. The same seed draws the same network "
        "again, another seed a different one; empty draws a fresh one every run"
    ),
    "synthetic_scale": (
        "Size of the synthetic box relative to the template's image; 1 is the same size. Vessel "
        "counts grow with it. Must be above 1 to surround the template's network"
    ),
    "synthetic_scale_axes": (
        "Which axes the synthetic box is enlarged along: x, y and z (default), or only x and y, "
        "keeping the template's depth for a wider field of the same tissue"
    ),
    "synthetic_extension_arteriole_density": (
        "Density of arterial trees in the box beyond the template's own size, relative to the "
        "template: 1 as dense, 0 none. Lower it if the image was framed round large vessels"
    ),
    "synthetic_extension_venule_density": (
        "Density of venous trees in the box beyond the template's own size, relative to the "
        "template: 1 as dense, 0 none. Lower it if the image was framed round large vessels"
    ),
    "synthetic_interior_dead_ends": (
        "Give the synthetic network as many dead ends inside the tissue as the template has. "
        "Off, its only dead ends are at the box's faces"
    ),
    "synthetic_open_ends": (
        "Whether synthetic vessels may leave the box. The default keeps every vessel inside, so "
        "blood enters and leaves only at inlets and outlets; the other cuts as many at the faces"
        " as the template has"
    ),
    "synthetic_bed_model": (
        "How the synthetic capillary bed is drawn: Delaunay (default) joins jittered lattice "
        "points; Voronoi runs capillaries along the edges of a Voronoi tessellation"
    ),
    "input_path": "The segmented image (a binary vessel mask, TIFF or HDF5) the network is built from",
    "segmentation_source": (
        "What the segmented vessels show, which sets how their widths are read for blood "
        "viscosity. Plasma column (default) for a luminal stain; Anatomical for a wall-to-wall "
        "endothelial stain"
    ),
    "segmented_image_downsample": (
        "Run on a coarser copy of the segmented image, merging blocks of 2, 4, 8 or 16 voxels "
        "per axis into one. Faster and lighter, but the thinnest vessels can be lost"
    ),
    "raw_image_path": (
        "The raw microscope image of the same field, used by ilastik, diameter measurement and "
        "the checks. Optional: without it the run works from the segmented image alone"
    ),
    "raw_image_channel": (
        "Which channel of a multi-channel raw image to read, counting from 0 (Fiji's C1 is 0). "
        "Leave empty for a single-channel file"
    ),
    "raw_image_stain": (
        "What the raw image stains, which decides how diameters are measured from it. Auto "
        "(default) tells from the image; or choose Plasma column (lumen) or Endothelial (vessel "
        "wall)"
    ),
    "raw_image_downsample": (
        "Read the raw image onto a coarser grid, averaging blocks of voxels. It is always "
        "brought onto the segmented image's grid; ilastik still reads the full file"
    ),
    "use_ilastik_segmentation": "Segment the raw image with a trained ilastik project instead of reading a segmented file",
    "ilastik_classifier_path": "The trained ilastik project (.ilp) that segments the raw image",
    "ilastik_vessel_label": (
        "Which ilastik label is vessel, by name or number (1 is the first label made). Every "
        "other label is background; the pre-run checks list the project's labels"
    ),
    "ilastik_executable": "The ilastik program to run without its window, by name or full path",
    "ilastik_timeout_seconds": (
        "Stop an ilastik run that takes longer than this many seconds. A hang usually means a "
        "bad project or unreadable image; raise it for very large images"
    ),
    "ilastik_reuse_existing_output": (
        "Reuse an earlier ilastik segmentation made from this unchanged image by the same "
        "project and label, instead of running ilastik again. Off segments again every run"
    ),
    "ilastik_output_dir": (
        "Folder ilastik's segmentations are written to, named after the raw image, one file per "
        "image and classifier"
    ),
    "ilastik_output_suffix": "File type of ilastik's segmentations: .tif or .h5",
    "voxel_size_override_xyz": (
        "Voxel size along x, y and z to use instead of the image's metadata: always under "
        "override, and under auto only when the metadata lacks it"
    ),
    "voxel_size_policy": (
        "Where the voxel size comes from. auto (default) uses the image metadata, else the "
        "override; override always uses the override; metadata_only stops if the metadata is "
        "incomplete"
    ),
    "image_axis_order": (
        "Which array axis is z, y and x in the input files. auto reads each file's own axis "
        "labels; zyx (default) is the usual order of a TIFF stack"
    ),
    "mid_run_postprocessing": (
        "Pause a panel run after Haemodynamics so the solved network can be edited by hand on "
        "the Post processing tab; its Continue button finishes the run"
    ),
    "use_memmap_loading": (
        "Keep large volumes on disk instead of in memory. Same result with much less RAM, but "
        "slower and needs free disk space the size of the image. Turn on for very large images"
    ),
    "memmap_directory": (
        "Folder for the low-RAM option's files, on a drive with room for an image-sized file. "
        "Empty uses the system temporary folder; set, later runs can reuse the loaded image"
    ),
    "segmentation_cleanup": "Clean up the segmented image before skeletonising; turn on to choose the clean-up steps",
    "segmentation_cleanup_fill_cavities": (
        "Fill small enclosed holes inside vessels before skeletonising, so a hollow vessel gets "
        "one centreline"
    ),
    "segmentation_cleanup_remove_whiskers": "Strip thin spikes, a voxel or two wide, sticking out of vessel surfaces before skeletonising",
    "segmentation_cleanup_whisker_radius_um": (
        "Surface spikes thinner than this radius are stripped. Keep it below the thinnest "
        "vessel's radius, or that vessel is stripped too"
    ),
    "segmentation_cleanup_split_narrow_necks": (
        "Cut the mask apart where two touching vessels were segmented as one blob joined by a "
        "narrow neck. A vessel of even width is never cut"
    ),
    "segmentation_cleanup_split_min_marker_separation_um": (
        "Least distance between the centres of two vessels before a narrow neck between them can"
        " be cut"
    ),
    "segmentation_cleanup_split_min_pinch_radius_ratio": (
        "Cut only where the neck is at most this fraction of the two vessels' typical radius. "
        "Lower cuts fewer necks"
    ),
    "segmentation_cleanup_split_min_body_radius_um": "Vessels with a typical radius below this are never split at a neck, being too thin to judge",
    "segmentation_cleanup_close_gaps": (
        "Close tiny gaps, about a voxel wide, within vessels with no shape test; a lighter "
        "version of \"Segmentation cleanup reconnect gaps\""
    ),
    "segmentation_cleanup_close_gaps_radius_um": (
        "Radius of the closing that seals small gaps in vessels. Larger seals wider gaps but can"
        " fuse vessels lying close together"
    ),
    "segmentation_cleanup_reconnect_gaps": (
        "Join pieces of the mask that look like one vessel continuing across a gap. Only "
        "aligned, tube-shaped pieces are joined, not any nearby blobs"
    ),
    "segmentation_cleanup_reconnect_max_bridge_distance_um": "Widest gap bridged between two pieces of one vessel",
    "segmentation_cleanup_reconnect_min_cylindricality": (
        "How tube-like both pieces must be to be joined, from 0 (blob) to 1 (straight tube). "
        "Higher joins fewer"
    ),
    "segmentation_cleanup_reconnect_max_axis_angle_degrees": (
        "Largest angle, in degrees, between the two pieces' directions for them to be joined. "
        "Lower joins only well-aligned pieces"
    ),
    "segmentation_cleanup_reconnect_min_facing_cosine": (
        "How directly each piece's end must point at the other to be joined, as a cosine: 1 "
        "straight at it, 0.85 within about 30 degrees"
    ),
    "segmentation_cleanup_reconnect_max_radius_ratio": (
        "Largest ratio between the two pieces' radii for them to be joined. Lower joins only "
        "pieces of similar width"
    ),
    "segmentation_cleanup_smooth_surfaces": "Smooth vessel surfaces in the mask before skeletonising, to reduce surface noise",
    "segmentation_cleanup_smooth_method": (
        "How surfaces are smoothed: gaussian (default) is fast but slightly thins narrow, curved"
        " vessels; morphological avoids that bias but gives coarser control"
    ),
    "segmentation_cleanup_smooth_sigma_um": (
        "Width of the blur used for smoothing. Larger smooths more, but erodes thin, curved "
        "vessels more"
    ),
    "segmentation_cleanup_smooth_morphological_radius_um": (
        "Radius of the shape used to smooth surfaces. Larger smooths bigger bumps, but also "
        "removes vessels thinner than this radius"
    ),
    "segmentation_cleanup_remove_small_volumes": "Remove small isolated pieces of the segmentation, usually noise, before skeletonising",
    "segmentation_cleanup_remove_small_min_volume_um3": (
        "Isolated pieces of the segmentation smaller than this volume are removed. Higher "
        "removes more"
    ),
    "crop_on_import": (
        "Analyse only a box cut from the image; every mask and raw image is cut to the same box."
        " Positions are then measured from the box's corner"
    ),
    "crop_method": (
        "How the box is given: size (default), a size placed at a location; or coordinates, the "
        "voxels of two opposite corners"
    ),
    "crop_size_xyz": "Size of the box along x, y and z. An axis shorter than this is kept whole",
    "crop_location": "Where the box sits in the x-y plane, as napari draws a slice: top is y = 0, left is x = 0",
    "crop_z_location": (
        "Which slices the box takes: first (from z = 0), centre (default, the middle of the "
        "stack) or last (the end of the stack)"
    ),
    "crop_start_xyz": (
        "Corner of the box with the smallest index on every axis, as x, y, z, as napari shows it"
        " under the cursor"
    ),
    "crop_end_xyz": (
        "Opposite corner of the box, as x, y, z; this voxel is kept. A corner past the image's "
        "edge stops at it"
    ),
    "expected_boundary_vessel_count": (
        "How many vessels you expect to cross the image edge, for the \"Check segmented image\" "
        "score; more than this lowers it. Empty assumes 2"
    ),
    "min_voxels_across_vessel_radius": (
        "Voxels across a vessel's radius that \"Check segmented image\" expects for full "
        "resolution marks; with fewer, diameters are less accurate. Empty uses 3"
    ),
    "min_acceptable_segmentation_quality": (
        "Warn before a run, never stopping it, when the segmentation's 0-10 quality score from "
        "\"Check segmented image\" is below this. Empty turns the warning off"
    ),
    # 2. Skeletonise
    "skeleton_thick_vessel_restrict_to_mask": (
        "Take thick vessels from the chosen vessel masks instead of judging them by width. off "
        "(default) judges by width alone; large, small or both use those masks"
    ),
    "skeleton_thick_vessel_restrict_to_mask_warn_below": (
        "Warn when less than this fraction of a chosen vessel mask lies on the segmented "
        "vessels, a sign the mask is misaligned or a different shape"
    ),
    "do_skeletonize": (
        "Skeletonise the segmented image. Off reuses the skeleton a previous run saved in the "
        "output folder"
    ),
    "skeleton_closing_radius": "Seal small gaps in the skeleton by thickening it and thinning it again. 0 turns this off",
    "skeleton_bridge_gap_size": (
        "Fill skeleton gaps up to this size by growing the skeleton, then thinning it again. 0 "
        "turns this off"
    ),
    "skeleton_min_branch_length": (
        "Skeleton pieces with fewer voxels than this are dropped, unless a bridge can join them "
        "to a larger piece"
    ),
    "skeleton_max_bridge_distance": (
        "Longest gap across which the end of a broken skeleton piece is joined to another piece."
        " 0 turns reconnection off"
    ),
    "skeleton_bridge_weight_by_segmentation": "Try a route through the segmented vessel before a straight line when joining skeleton pieces",
    "skeleton_bridge_z_distance_weight": (
        "Extra weight on the z part of a gap when judging whether it is short enough to bridge. "
        "Above 1 discourages joins mostly along z; 1 measures plain distance"
    ),
    "skeleton_bridge_min_facing_cosine": (
        "How directly a piece's end must point at what it joins, as a cosine: 0.5 allows up to "
        "60 degrees off, -1 any direction. Stops side-by-side vessels being cross-linked"
    ),
    "bridge_require_mask_support": (
        "Join gaps in the skeleton and graph only along routes through the segmented vessel, so "
        "vessels with background between them are never joined; short dropouts within one vessel"
        " still are"
    ),
    "bridge_max_background_gap_um": (
        "Longest single stretch of background a bridge may cross: a dropout in one vessel's "
        "segmentation, not tissue between two vessels"
    ),
    "bridge_min_mask_fraction": (
        "Least share of a bridge's length that must lie inside the segmented vessel, unless all "
        "the background it crosses fits within the longest background gap allowed"
    ),
    "skeleton_component_connectivity": (
        "Which voxels count as touching when skeleton pieces are counted: 3 (default) includes "
        "diagonal neighbours, 1 only voxels sharing a face"
    ),
    "skeleton_min_component_percent": (
        "Drop skeleton pieces holding less than this percentage of all skeleton voxels after "
        "clean-up. 0 keeps every piece"
    ),
    "skeleton_bundle_scan_size": (
        "Width of the window that finds dense tangles of skeleton, each collapsed into one "
        "junction with clean paths in and out"
    ),
    "skeleton_bundle_density_fraction": (
        "How full of skeleton a window must be to count as a tangle and be collapsed. Lower "
        "collapses more"
    ),
    "skeleton_bundle_max_connections_per_hub": "Most paths reconnected to the central junction of each collapsed tangle",
    "skeleton_bundle_hub_min_spacing": "Least spacing between the centres of collapsed tangles, so one tangle is not collapsed twice",
    "skeleton_mask_consistency_warn_below": (
        "Warn when less than this fraction of the segmented voxels lie within a vessel radius of"
        " the skeleton, a sign it misses parts of vessels"
    ),
    "missing_vessel_min_voxels": (
        "Segmented pieces smaller than this many voxels count as noise and are left out of the "
        "missing-vessel checks"
    ),
    "skeleton_missing_vessel_warn_below": (
        "Warn when less than this fraction of segmented vessels have any skeleton at all, a sign"
        " whole vessels were dropped. 1 warns at any loss"
    ),
    "use_thick_vessel_skeletonisation": (
        "Give wide vessels a single centreline tree instead of ordinary thinning, which can "
        "leave a sheet of strands inside them. Capillaries are thinned as usual"
    ),
    "skeleton_thick_vessel_min_radius_um": (
        "Radius above which a vessel is treated as thick and given a single centreline; narrower"
        " vessels are thinned as usual"
    ),
    "skeleton_fill_mask_holes_before_thickness": (
        "Fill enclosed holes in the mask before measuring vessel width, so a hollow vessel reads"
        " as thick. No effect on solid vessels"
    ),
    "skeleton_thick_vessel_wall_absorption_um": (
        "How far round a thick vessel's core its wall absorbs surface roughness; small vessels "
        "joining it lose this much length too. Empty uses half the thick-vessel radius threshold"
    ),
    "skeleton_thick_vessel_flake_filter_um": (
        "How far a thin vessel must reach beyond a thick vessel's wall to be kept, not dropped "
        "as a scrap of the wall. Lower keeps more short vessels, and more scraps"
    ),
    "skeleton_thick_vessel_max_bridge_radius_multiple": (
        "How far a thin vessel may be joined to a thick vessel's centreline, as a multiple of "
        "the thick vessel's radius there. Vessels farther away stay unjoined"
    ),
    "skeleton_thick_vessel_max_bridge_distance_um": (
        "Fixed limit on how far a thin vessel may be joined to a thick vessel's centreline; the "
        "tighter of this and the radius multiple applies. Empty uses the multiple alone"
    ),
    "skeleton_thick_vessel_bridge_radius_smoothing_um": (
        "How far along a thick vessel to look for its widest radius when judging a join, so a "
        "narrow waist does not block it. 0 reads a single point"
    ),
    "skeleton_thick_vessel_large_stack_tree": (
        "Draw thick-vessel centrelines with far less memory and no cap on their branches, for a "
        "very large stack. Use when one thick region spreads across the stack"
    ),
    "detect_thick_vessel_braiding": (
        "Warn when a thick vessel's centreline is still a tangle of several strands rather than "
        "one line. Never changes the skeleton"
    ),
    "thick_vessel_braid_factor_limit": (
        "Only warn about a thick vessel whose skeleton averages more than this many voxels per "
        "slice along it. 1 is a single clean centreline"
    ),
    "thick_vessel_braid_min_occupied_slices": (
        "Only warn about tangled thick-vessel centrelines at least this many slices long, so "
        "short stubs are ignored"
    ),
    "skeletonize_tile_large_components": (
        "Skeletonise a vessel piece too large for memory in overlapping slabs along z. Uses less"
        " memory; a wide enough overlap gives the same result in practice"
    ),
    "skeletonize_tile_max_voxels": (
        "Most voxels skeletonised in one slab while tiling; peak memory is about this many "
        "bytes. Lower uses less memory"
    ),
    "skeletonize_tile_halo_um": (
        "Overlap read from neighbouring slabs when tiling. Keep it well above the widest vessel "
        "radius, or seams can appear where slabs meet"
    ),
    # 3. Graph
    "do_graph_building": (
        "Build the vessel network from the skeleton. Off loads the network an earlier run saved "
        "in the output folder instead"
    ),
    "graph_reconnect_threshold": (
        "Largest gap bridged between the ends of network pieces while the network is built. "
        "Higher mends more breaks but may join vessels that are really separate"
    ),
    "final_orphan_reconnect_threshold": (
        "Largest gap bridged in the final pass that joins leftover loose pieces and dead ends to"
        " the network nearby"
    ),
    "min_stub_length": (
        "Shortest dead-end side branch kept, wherever it is not judged by the width of the "
        "vessel it leaves instead. Side branches ending at the image edge are kept"
    ),
    "min_stub_length_radius_multiple": (
        "Prune a dead-end side branch shorter than this many radii of the vessel it leaves, so "
        "spurs on wide vessels go while short capillary ends stay. 0 uses \"Min stub length\" "
        "everywhere"
    ),
    "remove_pieces_without_junction": (
        "Remove every piece of network with no junction, a lone vessel with two dead ends, "
        "whatever its length. Removes specks but also real vessels the segmentation cut off"
    ),
    "remove_interior_dead_ends": (
        "Remove every dead end inside the image back to its junction, then trace back segmented "
        "vessels that join the network at both ends. Can remove much of a fragmented network"
    ),
    "recovery_short_loop_keep_hole_over_um": (
        "In the clean-up after \"Remove interior dead ends\", keep a short loop round a tissue "
        "hole wider than this and break smaller ones. 0 breaks every short loop"
    ),
    "recover_uncovered_mask_vessels": (
        "Trace segmented vessels the network does not run through, such as a capillary the "
        "skeleton lost, and join them to the network through the segmentation"
    ),
    "recovery_min_region_volume_um3": (
        "Smallest piece of uncovered segmentation traced back as a vessel; smaller pieces are "
        "treated as specks"
    ),
    "recovery_min_length_um": "Shortest traced-back vessel kept, unless both its ends join the network",
    "facing_dead_end_max_gap_um": (
        "Join two dead ends pointing at each other across a gap in the segmentation up to this "
        "long, mending a vessel the segmentation broke. 0 turns this off"
    ),
    "remove_short_loops_under_um": (
        "Break every loop shorter than this all the way round, removing the side nearest the "
        "vessel wall, whatever the segmentation shows. 0 leaves short loops to the Short loops "
        "review list"
    ),
    "lumen_cleanup_max_rounds": (
        "Most rounds of the final clean-up, which repeats until nothing changes. Lower is faster"
        " on a large image but can leave doubled centrelines and loops"
    ),
    "short_loop_keep_hole_over_um": (
        "Leave a short loop to the Short loops review list instead of breaking it when the "
        "tissue hole it encircles is wider than this. 0 breaks every short loop"
    ),
    "lumen_cleanup_rewire_pairs": (
        "Where two centrelines run through one segmented vessel, cut out only the doubled "
        "stretch of the one nearer the wall and reconnect the rest, rather than deleting it "
        "whole"
    ),
    "lumen_cleanup_guard_rejoins": (
        "When removing a doubled centreline, never rejoin what is left in a way that closes a "
        "loop inside one vessel"
    ),
    "retrace_lost_skeleton": (
        "After the clean-up, draw back skeleton the network lost where it closes a loop round a "
        "tissue hole wider than \"Short loop keep hole over um\""
    ),
    "extend_dead_ends_to_faces": (
        "Draw a dead end that stops just short of the image edge, where its segmented vessel "
        "runs on, out to the edge so its open end sits there. Experimental"
    ),
    "trace_on_dead_ends": (
        "From a dead end where the segmented vessel carries on, draw the centreline on through "
        "the segmentation for up to 30 um, joining any vessel it reaches. Experimental"
    ),
    "join_dead_ends_to_vessel_sides": (
        "Join a dead end pointing at the side of another vessel, not a near neighbour, across a "
        "gap up to \"Facing dead end max gap um\". Experimental"
    ),
    "stub_face_protection": (
        "Which short side branches near the image edge escape pruning: threshold (default) any "
        "tip within its pruning length of the edge; leaves_image only those whose vessel "
        "actually leaves the image"
    ),
    "cluster_collapse_distance": (
        "Merge network nodes (junctions and vessel ends) closer than this into one. Higher "
        "tidies more clutter but can merge separate nearby junctions"
    ),
    "cluster_collapse_method": (
        "How nearby nodes are merged: distance_only (default) merges every node within the "
        "distance; direction_aware refuses merges that would make an implausible star-shaped "
        "junction"
    ),
    "cluster_collapse_max_radial_dispersion": (
        "Refuse a merge when the merged node's vessels would spread out this evenly or more, "
        "from 0 (perfectly evenly) to 1 (any spread). Higher refuses more merges"
    ),
    "cluster_collapse_persistence_search_multiple": (
        "How far to look for nodes to merge, as a multiple of \"Cluster collapse distance\". "
        "Wider lets each cluster find its own natural gap"
    ),
    "skeleton_graph_consistency_warn_below": "Warn in the log when the finished network follows less than this fraction of the skeleton",
    "graph_mask_consistency_warn_below": (
        "Warn in the log when less than this fraction of the segmented vessels lies within reach"
        " of a network centreline"
    ),
    "save_step_artifacts": (
        "Save the network and a picture of it after each graph-building step, to debug a bad "
        "network. Slows a large run greatly"
    ),
    "graph_missing_vessel_warn_below": (
        "Warn in the log when less than this fraction of the segmented vessels have any network "
        "centreline at all. 1 warns about any vessel missed"
    ),
    "thick_vessel_fill_centreline_gaps": (
        "Where a thick vessel's network centreline strays from or misses the centreline drawn "
        "through it, fill the gap from that centreline so vessels there can join it. Removes "
        "nothing"
    ),
    "thick_vessel_extend_centreline_ends": (
        "Where a thick vessel's centreline stops while its segmented vessel runs on, draw it on "
        "to the next vessel or dead end within 40 um, unless it reaches the image edge first"
    ),
    "thick_bridge_centreline_tolerance_um": (
        "How close a centreline must lie to a thick vessel's drawn centreline to count as that "
        "vessel when side vessels are joined to it, or 30% of its radius if larger"
    ),
    "smooth_centrelines": (
        "Smooth the voxel staircase out of each vessel's centreline, and measure its length from"
        " the smoothed curve"
    ),
    "centreline_smoothing_method": (
        "How centrelines are smoothed: taubin (default) gives the most accurate length; chaikin "
        "rounds the corners and moves the centreline least"
    ),
    "centreline_smoothing_iterations": (
        "Number of smoothing passes. Length accuracy improves up to about 15; beyond that the "
        "centreline slowly starts to bulge outwards"
    ),
    "centreline_max_deviation": (
        "How far a smoothed centreline may move from the skeleton before it is pulled back "
        "towards the original. Widened automatically for coarse voxels and wide vessels"
    ),
    "detect_cartwheel_hub_artifacts": (
        "Warn about junctions whose vessels radiate in too many directions to be real, a side "
        "effect of merging nearby nodes. Never changes the network"
    ),
    "cartwheel_hub_min_degree": (
        "Only warn about a junction joining at least this many vessels. Also used by the "
        "direction_aware merge method"
    ),
    "cartwheel_hub_max_radial_dispersion": (
        "How evenly a junction's vessels must spread out to be warned about, from 0 (perfectly "
        "evenly) to 1 (any spread). Higher warns about more junctions"
    ),
    "cartwheel_hub_tangent_length_um": (
        "How far along each vessel its direction from the junction is measured. Also used by the"
        " direction_aware merge method"
    ),
    # 4. Boundaries
    "automated_vessel_assignment": (
        "Use segmented arteriole and venule masks to choose inlets, outlets and where arterioles"
        " and venules hand over to capillaries, instead of setting them by hand"
    ),
    "use_large_vessel_masks": (
        "Use masks of the large arterioles and venules to place inlets and outlets. Ticking it "
        "fills the mask rows with matching files found beside the input image"
    ),
    "large_arteriole_mask_path": (
        "Segmented mask of the large arterioles. With no large venule mask, it is read as a mask"
        " of every large vessel"
    ),
    "large_venule_mask_path": (
        "Segmented mask of the large venules. Leave empty to read the large arteriole mask as a "
        "mask of every large vessel"
    ),
    "use_ilastik_large_vessel_segmentation": (
        "Segment the large arteriole and venule masks from the raw image with ilastik, instead "
        "of reading mask files"
    ),
    "ilastik_arteriole_classifier_path": (
        "Trained ilastik project (.ilp) that segments the large arterioles from the raw image. "
        "With no venule project, what it finds is read as every large vessel"
    ),
    "ilastik_arteriole_vessel_label": (
        "Which label of the large arteriole ilastik project is vessel: its name, or its number "
        "(1 is the first label made). Every other label is background"
    ),
    "ilastik_arteriole_channels": (
        "Channels of the raw image the large arteriole project was trained on, counting from 0, "
        "e.g. [1, 2]. Leave empty to give it every channel"
    ),
    "ilastik_venule_classifier_path": (
        "Trained ilastik project (.ilp) that segments the large venules from the raw image. "
        "Leave empty to read the arteriole project's output as every large vessel"
    ),
    "ilastik_venule_vessel_label": (
        "Which label of the large venule ilastik project is vessel: its name, or its number (1 "
        "is the first label made). Every other label is background"
    ),
    "ilastik_venule_channels": (
        "Channels of the raw image the large venule project was trained on, counting from 0, "
        "e.g. [2]. Leave empty to give it every channel"
    ),
    "inlets_outlets_from_vessel_masks": (
        "Make the network ends in the large arteriole mask its inlets and those in the large "
        "venule mask its outlets. Off: choose them with the inlet and outlet methods instead"
    ),
    "large_vessel_mask_dilation_microns": (
        "Grow the large arteriole and venule masks by this much as they are loaded, for every "
        "step that reads them. 0 keeps them as segmented"
    ),
    "large_vessel_assignment_max_dilation_microns": (
        "Grow the large masks in 5 um steps up to this far while choosing inlets and outlets, so"
        " network ends just outside them are found. 0 uses the masks as segmented"
    ),
    "large_vessel_min_component_volume_um3": (
        "Remove pieces of the large masks smaller than this, such as specks a classifier leaves."
        " 0 keeps every piece"
    ),
    "large_vessel_swap_minority_components": (
        "Where a large vessel is labelled partly arteriole and partly venule, relabel the "
        "smaller touching piece as the larger one's class, so it becomes one vessel"
    ),
    "large_vessel_swap_max_size_ratio": (
        "Relabel a piece only when it is at most this fraction of the size of the piece it "
        "touches. 1 allows any smaller piece; 0.5 only those half its size or less"
    ),
    "large_vessel_swap_min_contact_fraction": (
        "Relabel a piece only when at least this fraction of its surface touches the other "
        "class. Lower it to also catch short pieces joined end to end"
    ),
    "large_vessel_remove_small_opposite_attached_components": (
        "Remove tiny pieces of one large mask stuck to the other class's mask, which are usually"
        " mislabelled"
    ),
    "large_vessel_opposite_attached_max_component_volume_um3": "Largest piece stuck to the other large mask that is removed as mislabelled",
    "large_vessel_opposite_attached_max_distance_microns": (
        "How close a tiny piece must lie to the other large mask to count as stuck to it and be "
        "removed"
    ),
    "exclude_smaller_overlapping_volumes": (
        "Where the large arteriole and venule masks overlap, remove the shared voxels from the "
        "smaller vessel as the masks are loaded"
    ),
    "automated_vessel_assignment_enable_overlap_cleanup": (
        "Allow the overlap clean-up of the large masks while inlets and outlets are chosen: "
        "shared arteriole and venule voxels leave the smaller vessel. Off skips it, whatever "
        "fast mode says"
    ),
    "automated_vessel_assignment_fast_mode": (
        "Clean up the overlap between the large arteriole and venule masks once, before inlets "
        "and outlets are chosen"
    ),
    "automated_vessel_assignment_apply_overlap_cleanup_in_normal_mode": (
        "With fast mode off, still clean up the overlap between the large arteriole and venule "
        "masks while inlets and outlets are chosen"
    ),
    "automated_vessel_assignment_use_legacy_mode": (
        "Make each network end an inlet or outlet by which large mask reaches it first. Off "
        "scores each end by overlap, distance and neighbouring vessels, leaving uncertain ends "
        "unassigned"
    ),
    "automated_vessel_confidence_margin": (
        "How much better one large mask's score must be than the other's for an end to be "
        "assigned. Higher leaves more ends unassigned"
    ),
    "automated_vessel_min_confidence": (
        "Lowest confidence at which a network end is made an inlet or outlet. Higher leaves more"
        " ends unassigned"
    ),
    "automated_vessel_topology_penalty": (
        "How much the classes of an end's neighbouring vessels count towards making it an inlet "
        "or an outlet. 0 ignores them"
    ),
    "automated_vessel_quality_max_overlap_fraction": (
        "If the large arteriole and venule masks overlap by more than this, assign inlets and "
        "outlets cautiously: less mask growth and stricter confidence"
    ),
    "automated_vessel_quality_min_terminal_coverage": (
        "If less than this fraction of network ends lie in either large mask, assign inlets and "
        "outlets cautiously: less mask growth and stricter confidence"
    ),
    "automated_vessel_quality_max_component_count": (
        "If either large mask is in more pieces than this, assign inlets and outlets cautiously:"
        " less mask growth and stricter confidence"
    ),
    "automated_vessel_conservative_max_dilation_microns": (
        "Furthest the large masks are grown when they look poor and inlets and outlets are "
        "assigned cautiously"
    ),
    "write_fast_mode_preassignment_large_vessel_debug_3d_html": (
        "Save interactive 3D views (HTML) of the large masks before and after their overlap is "
        "cleaned up, to check the clean-up"
    ),
    "large_vessel_3d_volume_downsample_stride": (
        "Draw the masks in the vessel-mask 3D HTML views in blocks of this many voxels, keeping "
        "the files smaller. 1 draws every voxel"
    ),
    "cut_network_at_large_vessel_volumes": (
        "Remove the network inside the large arteriole and venule masks, leaving new ends at "
        "their surface to become inlets and outlets. Off keeps the large vessels"
    ),
    "remove_orphaned_branches_outside_large_vessel_volumes": "After cutting out the large vessels, remove small disconnected pieces of network left behind",
    "orphaned_branch_max_edge_count": (
        "Disconnected pieces with fewer vessels than this are removed after the large vessels "
        "are cut out"
    ),
    "assign_large_vessel_branch_orders": (
        "Keep the large vessels in the network with branch orders of their own, counted from "
        "where the image edge cuts them. Needs arteriole and venule boundaries too, from small "
        "masks or set by hand"
    ),
    "large_vessel_mask_min_overlap_fraction": (
        "Share of a vessel that must lie inside a large mask for it to be labelled a large "
        "arteriole or venule"
    ),
    "large_vessel_stump_min_component_fraction": (
        "Ignore pieces of a large mask smaller than this fraction of its largest piece when "
        "placing inlets and outlets at the image edge. 0 uses every piece"
    ),
    "use_small_vessel_masks_for_boundary_assignment": (
        "Use masks of the small arterioles and venules to find where they hand over to "
        "capillaries. Ticking it fills the mask rows with matching files found beside the input "
        "image"
    ),
    "small_arteriole_mask_path": (
        "Segmented mask of the small arterioles. With no small venule mask, it is read as a mask"
        " of every small vessel"
    ),
    "small_venule_mask_path": (
        "Segmented mask of the small venules. Leave empty to read the small arteriole mask as a "
        "mask of every small vessel"
    ),
    "use_ilastik_small_vessel_segmentation": (
        "Segment the small arteriole and venule masks from the raw image with ilastik, instead "
        "of reading mask files"
    ),
    "ilastik_small_arteriole_classifier_path": (
        "Trained ilastik project (.ilp) that segments the small arterioles from the raw image. "
        "With no venule project, what it finds is read as every small vessel"
    ),
    "ilastik_small_arteriole_vessel_label": (
        "Which label of the small arteriole ilastik project is vessel: its name, or its number "
        "(1 is the first label made). Every other label is background"
    ),
    "ilastik_small_arteriole_channels": (
        "Channels of the raw image the small arteriole project was trained on, counting from 0, "
        "e.g. [1, 2]. Leave empty to give it every channel"
    ),
    "ilastik_small_venule_classifier_path": (
        "Trained ilastik project (.ilp) that segments the small venules from the raw image. "
        "Leave empty to read the arteriole project's output as every small vessel"
    ),
    "ilastik_small_venule_vessel_label": (
        "Which label of the small venule ilastik project is vessel: its name, or its number (1 "
        "is the first label made). Every other label is background"
    ),
    "ilastik_small_venule_channels": (
        "Channels of the raw image the small venule project was trained on, counting from 0, "
        "e.g. [2]. Leave empty to give it every channel"
    ),
    "small_vessel_mask_min_overlap_fraction": (
        "Share of a vessel that must lie inside a small mask for it to be labelled a small "
        "arteriole or venule"
    ),
    "small_vessel_mask_dilation_microns": (
        "Grow the small masks step by step up to this far while labelling vessels, so vessels "
        "just outside them are reached. 0 uses the masks as segmented"
    ),
    "small_vessel_boundary_assignment_enable_overlap_cleanup": (
        "Allow the overlap clean-up of the small masks before vessels are labelled: shared "
        "arteriole and venule voxels leave the smaller vessel. Off skips it, whatever fast mode "
        "says"
    ),
    "small_vessel_boundary_assignment_fast_mode": (
        "Clean up the overlap between the small arteriole and venule masks once, before vessels "
        "are labelled"
    ),
    "small_vessel_boundary_assignment_apply_overlap_cleanup_in_normal_mode": (
        "With fast mode off, still clean up the overlap between the small arteriole and venule "
        "masks before vessels are labelled"
    ),
    "small_vessel_mask_continuity_enable": (
        "Bridge gaps in the small masks with short tubes, joining pieces of the same class "
        "(arteriole to arteriole, venule to venule) before vessels are labelled"
    ),
    "small_vessel_mask_continuity_allow_small_to_large": "Allow bridges from a small mask piece to the large mask of the same class",
    "small_vessel_mask_continuity_allow_small_to_small": "Allow bridges between two small mask pieces of the same class",
    "small_vessel_mask_continuity_enforce_cylinder_only": (
        "Bridge only tube-shaped pieces that point at each other, run the same way and have "
        "similar widths. Off checks only the bridge's length and path"
    ),
    "small_vessel_mask_continuity_min_cylindricality": (
        "How tube-like (0 to 1) a small mask piece must be to be bridged. Higher bridges only "
        "longer, straighter pieces"
    ),
    "small_vessel_mask_continuity_max_axis_angle_degrees": "Largest angle between the directions of two small mask pieces that can still be bridged",
    "small_vessel_mask_continuity_min_facing_cosine": (
        "How directly each piece must point at the other to be bridged, as a cosine: 1 head-on, "
        "0.82 within about 35 degrees"
    ),
    "small_vessel_mask_continuity_max_radius_ratio": (
        "Largest ratio between the widths of two small mask pieces that can still be bridged. 1 "
        "joins only equal widths"
    ),
    "small_vessel_mask_continuity_max_bridge_distance_microns": "Longest gap between two small mask pieces that a bridge may span",
    "small_vessel_mask_continuity_corridor_max_distance_microns": "Every point of a bridge must lie within this distance of mask of its own class",
    "small_vessel_mask_continuity_opposite_exclusion_distance_microns": "Refuse a bridge that comes within this distance of the other class's mask",
    "small_vessel_tangential_redefinition_enable": (
        "Relabel a small arteriole or venule piece that runs alongside a large vessel as that "
        "vessel's class, and flip small pieces lying between two of the other class"
    ),
    "small_vessel_tangential_redefinition_max_contact_distance_microns": "How far from a large vessel a small piece is examined to judge whether it runs alongside it",
    "small_vessel_tangential_redefinition_touch_distance_microns": "How close a small piece must come to a large vessel to count as touching it",
    "small_vessel_tangential_redefinition_tangency_cosine_max": (
        "How steeply a small piece may meet a large vessel and still count as running alongside "
        "it, as a cosine: 0 only parallel, 0.35 within about 20 degrees"
    ),
    "small_vessel_tangential_redefinition_min_contact_fraction": (
        "Relabel a small piece only when at least this fraction of it touches the large vessel, "
        "so one brief contact does not relabel a whole tree"
    ),
    "small_vessel_tangential_redefinition_margin": (
        "When a small piece runs alongside both a large arteriole and a large venule, how "
        "clearly one contact must win before the piece takes its class"
    ),
    "small_vessel_tangential_redefinition_parallel_workers": (
        "Threads used to judge small pieces against the large vessels. More is faster on a "
        "many-core machine; 0 or 1 runs one at a time"
    ),
    "small_vessel_sandwiched_reassignment_enable": (
        "Flip a small piece lying in line between two pieces of the other class, when it is "
        "smaller than the two together"
    ),
    "small_vessel_sandwiched_max_gap_microns": "Largest gap between a sandwiched piece's end and the piece beyond it",
    "small_vessel_sandwiched_min_facing_cosine": (
        "How directly the piece beyond each end must lie ahead of it, as a cosine: 1 straight "
        "ahead, 0.82 within about 35 degrees"
    ),
    "small_vessel_sandwiched_max_axis_angle_degrees": (
        "Largest angle between a sandwiched piece and each neighbour near it, so a vessel it "
        "only meets side-on does not count"
    ),
    "use_gpu_mask_continuity_acceleration": (
        "Speed up the small-mask gap bridging on a graphics card (needs CuPy installed). Without"
        " one it runs on the processor as usual"
    ),
    "small_vessel_boundary_fallback_to_hop_distance": (
        "For each inlet or outlet whose vessel the masks never reach, also mark the nodes a set "
        "number of vessels away as arteriole or venule boundaries"
    ),
    "small_vessel_boundary_fallback_hop_distance": "How many vessels away from an uncovered inlet or outlet the added boundary nodes lie",
    "small_vessel_min_component_volume_um3": (
        "Remove pieces of the small masks smaller than this, such as specks a classifier leaves."
        " 0 keeps every piece"
    ),
    "small_vessel_swap_minority_components": (
        "Where a small vessel is labelled partly arteriole and partly venule, relabel the "
        "smaller touching piece as the larger one's class, so it becomes one vessel"
    ),
    "small_vessel_swap_max_size_ratio": (
        "Relabel a piece only when it is at most this fraction of the size of the piece it "
        "touches. 1 allows any smaller piece; 0.5 only those half its size or less"
    ),
    "small_vessel_swap_min_contact_fraction": (
        "Relabel a piece only when at least this fraction of its surface touches the other "
        "class. Lower it to also catch short pieces joined end to end"
    ),
    "write_small_vessel_boundary_labelling_3d_html": (
        "Save an interactive 3D view (HTML) of the small masks, the vessels they label and the "
        "arteriole and venule boundary nodes"
    ),
    "classify_vessels_by_size": (
        "Class each vessel wide, mid or narrow by its measured width instead of by vessel masks."
        " No vessel is called arteriole or venule; narrow ones are treated as capillaries"
    ),
    "vessel_size_mid_min_diameter_um": (
        "Narrowest vessel classed mid; anything narrower is narrow. Compared with widths as "
        "measured, so a wall-to-wall threshold does not suit a plasma-column measurement"
    ),
    "vessel_size_wide_min_diameter_um": (
        "Narrowest vessel classed wide; must be above the mid threshold. Compared with widths as"
        " measured, wall to wall or plasma column"
    ),
    "inlet_node_selection_method": (
        "How inlets are chosen. edge_percent (default) takes the network ends at the start of "
        "the boundary axis; coordinates, volume or node_ids take points, boxes or nodes you pick"
    ),
    "outlet_node_selection_method": (
        "How outlets are chosen. edge_percent (default) takes the network ends at the end of the"
        " boundary axis; coordinates, volume or node_ids take points, boxes or nodes you pick"
    ),
    "arteriole_boundary_selection_method": (
        "How arteriole boundary nodes (where arterioles hand over to capillaries) are picked: "
        "coordinates (default), volume or node_ids, from that method's list. With the list "
        "empty, none are set"
    ),
    "venule_boundary_selection_method": (
        "How venule boundary nodes (where capillaries hand over to venules) are picked: "
        "coordinates (default), volume or node_ids, from that method's list. With the list "
        "empty, none are set"
    ),
    "large_vessel_inlet_node_selection_method": (
        "How the large vessels' inlets are chosen. mask_stump (default) places them where the "
        "image edge cuts the large arteriole mask; any other method overrides that, as for "
        "ordinary inlets"
    ),
    "large_vessel_outlet_node_selection_method": (
        "How the large vessels' outlets are chosen. mask_stump (default) places them where the "
        "image edge cuts the large venule mask; any other method overrides that, as for ordinary"
        " outlets"
    ),
    "boundary_axis": "Image axis along which edge_percent separates inlets from outlets: 0 is z, 1 is y, 2 is x",
    "boundary_first_percent": (
        "Share of the network's span, from its start along the boundary axis, whose ends become "
        "inlets with edge_percent"
    ),
    "boundary_last_percent": (
        "Share of the network's span, back from its end along the boundary axis, whose ends "
        "become outlets with edge_percent"
    ),
    "boundary_distance_from_inlet_node": (
        "With degree_1_from_inlet, take only network ends further than this in a straight line "
        "from every inlet"
    ),
    "boundary_open_end_max_distance_um": (
        "With edge_percent or degree_1_from_inlet, take only ends this close to an image face, "
        "where the image cut a vessel. Leave empty to allow any end"
    ),
    "boundary_open_end_test": (
        "How edge_percent and degree_1_from_inlet tell an end the image cut. distance (default):"
        " near an image face. leaves_image: graph building judged that it leaves the image"
    ),
    "inlet_node_coordinates": "Points as (z, y, x) whose nearest network ends become inlets, with the coordinates method",
    "outlet_node_coordinates": "Points as (z, y, x) whose nearest network ends become outlets, with the coordinates method",
    "arteriole_boundary_node_coordinates": (
        "Points as (z, y, x) whose nearest network ends become arteriole boundary nodes, with "
        "the coordinates method"
    ),
    "venule_boundary_node_coordinates": (
        "Points as (z, y, x) whose nearest network ends become venule boundary nodes, with the "
        "coordinates method"
    ),
    "large_vessel_inlet_node_coordinates": (
        "Points as (z, y, x) whose nearest network ends become the large vessels' inlets, with "
        "the coordinates method"
    ),
    "large_vessel_outlet_node_coordinates": (
        "Points as (z, y, x) whose nearest network ends become the large vessels' outlets, with "
        "the coordinates method"
    ),
    "inlet_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become inlets, with "
        "the volume method"
    ),
    "outlet_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become outlets, with "
        "the volume method"
    ),
    "arteriole_boundary_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become arteriole "
        "boundary nodes, with the volume method"
    ),
    "venule_boundary_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become venule "
        "boundary nodes, with the volume method"
    ),
    "large_vessel_inlet_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become the large "
        "vessels' inlets, with the volume method"
    ),
    "large_vessel_outlet_node_volumes": (
        "Boxes, each two opposite corners as (z, y, x), whose network ends become the large "
        "vessels' outlets, with the volume method"
    ),
    "inlet_node_ids": (
        "Network node IDs to use as inlets, exactly, with the node_ids method. Pick them by "
        "clicking nodes; rebuilding the network renumbers them"
    ),
    "outlet_node_ids": (
        "Network node IDs to use as outlets, exactly, with the node_ids method. Pick them by "
        "clicking nodes; rebuilding the network renumbers them"
    ),
    "arteriole_boundary_node_ids": (
        "Network node IDs to use as arteriole boundary nodes, exactly, with the node_ids method."
        " Pick them by clicking nodes; rebuilding the network renumbers them"
    ),
    "venule_boundary_node_ids": (
        "Network node IDs to use as venule boundary nodes, exactly, with the node_ids method. "
        "Pick them by clicking nodes; rebuilding the network renumbers them"
    ),
    "large_vessel_inlet_node_ids": (
        "Network node IDs to use as the large vessels' inlets, exactly, with the node_ids "
        "method. Pick them by clicking nodes; rebuilding the network renumbers them"
    ),
    "large_vessel_outlet_node_ids": (
        "Network node IDs to use as the large vessels' outlets, exactly, with the node_ids "
        "method. Pick them by clicking nodes; rebuilding the network renumbers them"
    ),
    "inlet_nodes": "The inlet nodes the run chose, filled in by the run. Anything entered here is replaced",
    "outlet_nodes": "The outlet nodes the run chose, filled in by the run. Anything entered here is replaced",
    "arteriole_boundary_nodes": (
        "The arteriole boundary nodes the run chose, filled in by the run. Anything entered here"
        " is replaced"
    ),
    "venule_boundary_nodes": (
        "The venule boundary nodes the run chose, filled in by the run. Anything entered here is"
        " replaced"
    ),
    "large_vessel_inlet_nodes": (
        "The large vessels' inlet nodes the run chose, filled in by the run. Anything entered "
        "here is replaced"
    ),
    "large_vessel_outlet_nodes": (
        "The large vessels' outlet nodes the run chose, filled in by the run. Anything entered "
        "here is replaced"
    ),
    "large_arteriole_boundary_nodes": (
        "Nodes where the large arterioles hand over to smaller arterioles, filled in by the run."
        " Anything entered here is replaced"
    ),
    "large_venule_boundary_nodes": (
        "Nodes where smaller venules hand over to the large venules, filled in by the run. "
        "Anything entered here is replaced"
    ),
    # 5. Diameters
    "strict_branch_order_assignment": (
        "Stop the run when arteriole and venule boundaries were expected, from small-vessel "
        "masks or a manual pick, but none were found, rather than labelling capillary orders "
        "only"
    ),
    "all_diams_const": (
        "Give every branch order the same fallback width, the \"Default diameter\". Turn off to "
        "set fallback widths per branch order"
    ),
    "manual_capillary_diameter_by_branch_order": (
        "Fallback capillary widths by branch-order label (B01, B02 ...), for vessels without a "
        "measured width. Orders not listed use the \"Default diameter\""
    ),
    "manual_arteriole_diameter_by_branch_order": (
        "Fallback arteriole widths by branch-order label (Art1, Art2 ...), for vessels without a"
        " measured width. Orders not listed use the B01 capillary width"
    ),
    "manual_venule_diameter_by_branch_order": (
        "Fallback venule widths by branch-order label (Ven1, Ven2 ...), for vessels without a "
        "measured width. Orders not listed use the B01 capillary width"
    ),
    "manual_large_arteriole_diameter_by_branch_order": (
        "Fallback large-arteriole widths by branch-order label (Large_Art1 ...), for vessels "
        "without a measured width. Orders not listed use the \"Default diameter\""
    ),
    "manual_large_venule_diameter_by_branch_order": (
        "Fallback large-venule widths by branch-order label (Large_Ven1 ...), for vessels "
        "without a measured width. Orders not listed use the \"Default diameter\""
    ),
    "diameter_by_branch_order": (
        "A complete table of fallback widths by branch-order label, replacing the one built from"
        " the per-class rows. Leave empty to build it from them"
    ),
    "max_branch_order": (
        "Highest branch order the diameter and constriction tables are built to. Vessels of "
        "higher orders take the highest order's width"
    ),
    "default_diameter": (
        "Width a vessel falls back to when nothing measured it and the branch-order table lists "
        "no width for it. With \"All diams const\" on, every order uses it"
    ),
    "diameter_class_median_min_edges": (
        "Give an unmeasured vessel the median measured width of vessels with its branch-order "
        "label, if at least this many were measured. 0 goes straight to the diameter table"
    ),
    "measure_diameters_from_raw_image": (
        "Measure each vessel's width from the raw image: by FWHM (full width at half maximum of "
        "its brightness profile) for a lumen stain, or inside the wall ring of an endothelial "
        "stain"
    ),
    "do_fwhm_measurement": (
        "Measure FWHM widths from the raw image. Turn off to keep the widths a loaded run "
        "already has instead of measuring again"
    ),
    "fwhm_sample_spacing_along_edge_um": (
        "Distance between width readings along each vessel; the mask widths use it too. Smaller "
        "takes more readings but is slower"
    ),
    "fwhm_transverse_profile_step_um": (
        "Spacing of brightness samples along each line drawn across a vessel. Smaller is finer "
        "but slower"
    ),
    "fwhm_transverse_half_extent_um": (
        "How far each line across a vessel first reaches either side of the centreline. It is "
        "lengthened automatically when the vessel proves wider"
    ),
    "fwhm_diameter_guess_um": (
        "Starting width for the fit on vessels with no stored width to start from. Leave empty "
        "to size the first line by \"Fwhm transverse half extent um\" alone"
    ),
    "fwhm_diameter_guess_edge_attribute": (
        "Name of a width already stored on each vessel to start its fit from; the default is its"
        " mask width, when measured. Clear it to use one guess for every vessel"
    ),
    "fwhm_min_total_extent_multiplier": (
        "Shortest line across a vessel kept, as a multiple of its fitted width, so background "
        "shows either side. Shorter lines are dropped unless another vessel cut them short"
    ),
    "fwhm_max_transverse_widen_passes": (
        "Most times a line across a vessel is lengthened when the fit finds the vessel wider "
        "than the line allows. 0 never lengthens it"
    ),
    "fwhm_background_label": (
        "Internal code for empty voxels in the map of centrelines that lines across vessels are "
        "checked against. Must differ from the junction code; rarely needs changing"
    ),
    "fwhm_junction_label": (
        "Internal code for voxels two centrelines share in the map lines across vessels are "
        "checked against. Must differ from the background code; rarely needs changing"
    ),
    "fwhm_allow_junction_crossing": (
        "Let a line across a vessel run on through points where centrelines meet, instead of "
        "stopping there"
    ),
    "fwhm_allow_crossing_other_edges": (
        "Let a line across a vessel run past another centreline instead of stopping there, so a "
        "wide vessel drawn as several centrelines is not read too narrow"
    ),
    "fwhm_profile_baseline_mode": (
        "How each line's background brightness is estimated. wings (default) uses the line's two"
        " ends; percentile uses a low percentile of the whole line"
    ),
    "fwhm_profile_baseline_wing_fraction": "Share of the line at each end used to estimate the background brightness",
    "fwhm_constrain_fitted_baseline": (
        "Keep the fitted background brightness close to its first estimate instead of letting "
        "the fit move it freely"
    ),
    "fwhm_baseline_constraint_half_width_ptp": (
        "How far the fitted background may move from its first estimate either way, as a "
        "fraction of the line's brightness range"
    ),
    "fwhm_clip_profile_to_single_vessel": (
        "Cut each brightness profile at the dip before a neighbouring vessel, so only this "
        "vessel's peak is fitted"
    ),
    "fwhm_clip_min_drop_fraction_of_center": (
        "How far brightness must fall from the vessel's centre, as a fraction of the centre "
        "value, before a rise beyond it counts as a neighbouring vessel"
    ),
    "fwhm_clip_re_rise_fraction_of_center": (
        "How much brightness must rise again after the dip, as a fraction of the centre value, "
        "for the profile to be cut there. Lower cuts at smaller bumps"
    ),
    "fwhm_branch_endpoint_exclusion_um": (
        "Take no readings this close to a vessel's ends where it meets other vessels, where "
        "widths are unreliable. Short vessels keep a central stretch"
    ),
    "fwhm_junction_proximity_exclusion_um": (
        "Take no readings this close to points where the vessel's centreline meets or crosses "
        "another's"
    ),
    "fwhm_enforce_same_edge_locality": (
        "Stop a line across a vessel where it reaches a distant part of the same vessel, as "
        "where the vessel bends back on itself"
    ),
    "fwhm_same_edge_arc_window_um": (
        "How far along the vessel a point must lie from a reading to count as a distant part "
        "of the same vessel. Empty sizes it by each vessel's width"
    ),
    "fwhm_same_edge_arc_window_multiplier": (
        "Size of the same-vessel window in vessel widths, used while \"Fwhm same edge arc "
        "window um\" is left empty"
    ),
    "fwhm_same_edge_arc_window_min_um": (
        "Smallest same-vessel window when it is sized by vessel width, used while \"Fwhm same "
        "edge arc window um\" is left empty"
    ),
    "fwhm_cap_half_extent_by_nonlocal_same_edge_distance": (
        "Shorten a line across a vessel where another part of the same vessel lies close by, as "
        "at a tight bend, so the line does not reach it"
    ),
    "fwhm_nonlocal_same_edge_arc_separation_um": (
        "How far apart along the vessel two points must be to count as different parts of it, "
        "when shortening lines at bends"
    ),
    "fwhm_nonlocal_same_edge_half_extent_factor": (
        "Each side of a line is cut to this fraction of the distance to the nearest other part "
        "of the same vessel"
    ),
    "fwhm_reject_samples_with_center_offset": (
        "Discard readings whose fitted vessel centre lies too far from the centreline, a sign "
        "the line caught something else"
    ),
    "fwhm_max_fit_center_offset_um": (
        "How far a reading's fitted centre may stray from the centreline before it is discarded."
        " Wide vessels may stray further, as their width fraction allows"
    ),
    "fwhm_max_fit_center_offset_fraction_of_diameter": (
        "Let a fitted centre stray this fraction of the vessel's estimated width from the "
        "centreline, where that is more than the fixed distance"
    ),
    "fwhm_reject_samples_with_low_fit_r2": "Discard readings whose fitted curve matches the brightness profile poorly",
    "fwhm_min_fit_r2": "Lowest fit quality (R-squared, 0 to 1) a reading needs to be kept. Higher is stricter",
    "fwhm_profile_model": (
        "How each brightness profile is fitted. blurred_lumen (default) fits a filled vessel "
        "seen through the microscope's blur; gaussian fits a bell curve, which reads wide "
        "vessels slightly narrow"
    ),
    "fwhm_longitudinal_average_um": (
        "Length of vessel each line across it is averaged over before fitting, which smooths "
        "noise on a dim channel. 0 reads a single line"
    ),
    "fwhm_stop_at_other_vessels_in_mask": (
        "End a line across a vessel where it enters another vessel of the segmentation after "
        "crossing background, so the neighbour is not read as part of this one"
    ),
    "fwhm_min_diameter_pixels": (
        "Narrowest reading kept: anything smaller cannot be resolved and is usually a fit to a "
        "noise spike. 0 keeps every reading"
    ),
    "fwhm_reject_samples_with_plateau_shape": (
        "Discard readings whose profile is a flat top rather than a peak, as from a saturated "
        "detector or a probability map. Leave off for plasma-filled lumens, which are "
        "flat-topped"
    ),
    "fwhm_max_plateau_shape_ratio": (
        "How flat-topped a profile may be before it is discarded: its width at 80% height as a "
        "fraction of its half-height width. Lower discards more"
    ),
    "fwhm_transverse_sampling_mode": (
        "Which way lines cross a vessel. in_plane_yx (default) stays in each image slice, "
        "reading steeply tilted vessels too wide; true_3d_perpendicular crosses at right angles "
        "to the vessel in 3D"
    ),
    "fwhm_warn_out_of_plane_tangent_fraction": (
        "While lines stay in each slice, flag vessels whose direction tilts out of the slice by "
        "more than this fraction, as their widths may read too wide. Widths are unchanged"
    ),
    "fwhm_edge_diameter_aggregation": (
        "How a vessel's accepted readings become one width: median (default) resists a single "
        "bad reading; mean averages them all. The mask widths use it too"
    ),
    "fwhm_min_accepted_samples": (
        "Fewest good readings a vessel needs before its FWHM width is used. More guards against "
        "fitting one bright speck; 1 accepts a single reading"
    ),
    "fwhm_decoy_check": (
        "After measuring, re-run FWHM on copies of measured vessels moved into empty tissue, "
        "report how often it finds a width there, and flag real widths in that range"
    ),
    "fwhm_demote_flagged_edges": (
        "Do not model a vessel with a flagged FWHM width (in the empty-tissue range, or far from"
        " its mask width); use the next diameter source. The width stays for review"
    ),
    "fwhm_fix_blur_to_image_psf": (
        "Use one blur for the whole image, from the raw-section blur rows or estimated from wide"
        " vessels, instead of fitting it in every profile"
    ),
    "fwhm_decoy_check_sample_size": (
        "How many measured vessels the decoy check copies into empty tissue. More gives a firmer"
        " estimate but takes longer"
    ),
    "use_raw_section_fallback": (
        "Where FWHM fails, fit the vessel's whole raw cross-section before falling back to the "
        "mask. Reads wide and z-running vessels, but can read the halo on speckled images"
    ),
    "raw_section_min_lumen_contrast": (
        "How far a cross-section's fitted lumen must stand above the background, in standard "
        "deviations of its texture, to count. Each vessel needs two such readings"
    ),
    "raw_section_psf_sigma_xy_um": (
        "The raw image's blur across each slice, as a Gaussian's standard deviation. Set it with"
        " the z blur, or leave both empty to estimate them from wide vessels"
    ),
    "raw_section_psf_sigma_z_um": (
        "The raw image's blur along z, as a Gaussian's standard deviation. Set it with the "
        "in-slice blur, or leave both empty to estimate them from wide vessels"
    ),
    "endothelial_min_ring_contrast": (
        "How clearly the stained wall must stand out above the lumen, in multiples of the fit's "
        "noise, for a reading to count. Each vessel needs two such readings"
    ),
    "endothelial_wall_thickness_um": (
        "Thickness of the stained vessel wall. Leave empty to estimate it from wide vessels; set"
        " it if known, since an error here shifts every internal width by about as much"
    ),
    "endothelial_workers": (
        "How many vessels are measured at once. 0 chooses automatically, up to 8, leaving a "
        "processor core free; the widths are the same whatever the number"
    ),
    "endothelial_psf_sigma_xy_um": (
        "The endothelial channel's blur across each slice, as a Gaussian's standard deviation. "
        "Set it with the z blur, or leave both empty to estimate them from wide vessels"
    ),
    "endothelial_psf_sigma_z_um": (
        "The endothelial channel's blur along z, as a Gaussian's standard deviation. Set it with"
        " the in-slice blur, or leave both empty to estimate them from wide vessels"
    ),
    "use_edt_diameter_crosscheck": (
        "Measure each vessel's width in the segmentation mask, to check the raw-image widths and"
        " to stand in where they fail. Without raw-image measurement, it can give every vessel "
        "its width"
    ),
    "edt_diameter_prefer_over_table_on_fwhm_failure": (
        "Where the raw image gives a vessel no width, use its mask width rather than the median "
        "of similar vessels or the branch-order table"
    ),
    "edt_diameter_method": (
        "How a width is read off the mask. cross_section (default) turns the area of the "
        "vessel's cross-section into a circle's diameter; inscribed_radius doubles the largest "
        "fitting radius, less accurate"
    ),
    "edt_min_resolvable_diameter_um": (
        "Ignore mask widths narrower than this: a sliver of mask a voxel or so thick, not the "
        "vessel's real width. 0 keeps every reading"
    ),
    "edt_junction_proximity_exclusion_um": (
        "Take no mask readings this close to a vessel's ends where it meets other vessels, where"
        " the mask bulges. The endothelial ring fit uses it too"
    ),
    "edt_fwhm_disagreement_warn_ratio": (
        "Flag a vessel's FWHM width as doubtful when it differs from its mask width by this "
        "ratio or more, either way"
    ),
    "custom_edges": (
        "Vessels, each given as a pair of node IDs, whose resistance is computed for a fixed 6 "
        "um width whatever width was measured"
    ),
    # 6. Haemodynamics
    "inlet_p_bc": "Blood pressure held at every inlet node in the flow solve",
    "outlet_p_bc": "Blood pressure held at every outlet node in the flow solve",
    "run_haemodynamics": (
        "Run the haemodynamics stage: give each vessel a resistance and solve the blood flow and"
        " pressures through the network"
    ),
    "do_equiv_resistance_calculation": (
        "Compute the network's resistance between two nodes: the first inlet and first outlet, "
        "unless Post processing picked two others"
    ),
    "haemodynamics_solver": (
        "How each flow solve handles the network. Dense solver (default) suits small networks; "
        "Sparse solver gives the same flows far faster and with far less memory on large ones"
    ),
    "equivalent_resistance_solver": (
        "How the resistance between the two chosen nodes is found. Eigendecomposition solver "
        "(default) is slow and memory-hungry on large networks; Sparse solver gives the same "
        "answer much faster"
    ),
    "viscosity_law": (
        "Law that sets blood viscosity from vessel width. pries (default) is the standard "
        "empirical law for blood in small vessels; constant uses plasma viscosity everywhere. "
        "Resistances differ between laws"
    ),
    "pries_viscosity_form": (
        "Which form of the Pries law each vessel uses. Auto (default) picks in vitro for a "
        "plasma-column diameter, in vivo for a wall-to-wall one; the others force one form"
    ),
    "haematocrit": (
        "Share of the flowing blood that is red cells, read by the Pries viscosity law and the "
        "red-cell predictions. With a calculated haematocrit distribution, it is the share "
        "entering at the inlets"
    ),
    "red_cell_volume_fl": (
        "Mean red-cell volume, used to count red cells in the flow predictions: about 50 for "
        "mouse, 90 for human. It changes no resistance or flow"
    ),
    "haematocrit_model": (
        "How haematocrit is set per vessel. Fixed haematocrit (default) is one value everywhere;"
        " the calculated distribution splits red cells unevenly at branches, re-solving until it"
        " settles. Only the Pries law uses it"
    ),
    "haematocrit_junction_rule": (
        "How red cells divide at junctions other than one vessel splitting in two. No phase "
        "separation (default) gives every outflow the mixed inflow's haematocrit; the others "
        "treat the junction as bifurcations"
    ),
    "haematocrit_split_connector_length_um": (
        "Length of the short connector vessels added when 4+ vessel junctions are split. 0 makes"
        " each as long as the junction's vessels are wide"
    ),
    "haematocrit_distribution_max_iterations": (
        "Most flow re-solves before the haematocrit distribution stops trying to settle. Raise "
        "it if the log warns that it did not converge"
    ),
    "haematocrit_distribution_tolerance": (
        "Largest change in any vessel's haematocrit between passes that counts as settled. Lower"
        " is more exact but takes more passes"
    ),
    "boundary_handling": (
        "What happens to vessels no inlet-to-outlet route runs along, which the flow solve "
        "cannot answer. Leave unsolved (default) keeps them, greyed in the viewer; Remove "
        "disconnected deletes them"
    ),
    "open_end_model": (
        "How vessels running out of the image are treated in the flow solve. estimated (default)"
        " lets them feed or drain the network; closed lets no blood through them"
    ),
    "open_end_terminal_resistance_multiple": (
        "How many times its own vessel's resistance each open end drains through to the outlet "
        "pressure. Higher lets less blood out"
    ),
    "open_end_estimate_shear_weight": (
        "How much keeping wall shear stress near its target counts, against keeping pressure "
        "near its target, when estimating open-end pressures. Higher favours the shear target"
    ),
    "open_end_estimate_pressure_target_pa": (
        "Pressure the open-end estimate keeps the network near. Leave empty for the mean "
        "pressure the solve gives with open ends closed"
    ),
    "open_end_estimate_shear_target_pa": (
        "Wall shear stress the open-end estimate keeps each vessel near. Leave empty for the "
        "median the solve gives with open ends closed"
    ),
    # 8. Perturbations
    "do_pericyte_construction": (
        "Has no effect: pericyte constrictions come only from the pericyte perturbation types on"
        " this tab"
    ),
    "constriction_length_um": "Length along the vessel of each pericyte constriction or dilation site",
    "constriction_spacing_um": (
        "Distance between neighbouring pericyte sites along each capillary, centre to centre, "
        "when sites are spaced regularly rather than read from a mask"
    ),
    "use_pericyte_mask_constriction": (
        "Place pericyte constriction sites at the pericytes of a segmented mask, rather than at "
        "a regular spacing along each capillary"
    ),
    "pericyte_mask_path": (
        "Segmented pericyte mask, binary or labelled; each separate blob is one pericyte. Filled"
        " in from a file named \"pericyte\" beside the input image when one is found"
    ),
    "pericyte_mask_h5_dataset_name": "Name of the dataset holding the pericyte mask inside an H5 file; ignored for TIFF masks",
    "pericyte_max_assignment_distance_um": "Ignore a pericyte lying further than this from the nearest capillary centreline",
    "pericyte_min_diameter_um": (
        "Smallest blob in the pericyte mask counted as a pericyte, measured as the width of a "
        "sphere of the same volume"
    ),
    "pericyte_max_diameter_um": (
        "Largest blob in the pericyte mask counted as a pericyte, measured as the width of a "
        "sphere of the same volume"
    ),
    "use_probabilistic_pericyte_constriction": "Let only a random share of pericyte sites constrict or dilate, rather than every site",
    "pericyte_constriction_probability": "Share of pericyte sites that constrict or dilate, chosen at random: 0 none, 1 all",
    "pericyte_constriction_seed": (
        "Random seed that picks which pericyte sites act, so a run repeats exactly; another seed"
        " picks a different set"
    ),
    "run_pericyte_resistance_comparison": (
        "Compare network resistance with and without pericyte constriction, for the "
        "haemodynamics library only. Has no effect in a pipeline run; use a pericyte "
        "perturbation instead"
    ),
    "pericyte_comparison_baseline_value": (
        "Constriction factor for the baseline side of the pericyte resistance comparison; 1 "
        "means no narrowing"
    ),
    "pericyte_comparison_constricted_value": (
        "Constriction factor for the constricted side of the pericyte resistance comparison; 0.8"
        " narrows each site by 20%"
    ),
    "reuse_comparison_pericyte_cohort_for_main_run": (
        "Constrict the same randomly chosen pericytes in the main run as in the pericyte "
        "resistance comparison"
    ),
    "pericyte_constriction_factor": (
        "How each pericyte site changes the vessel's diameter: 1 leaves it unchanged, below 1 "
        "narrows it (0.8 is 20% narrower), above 1 widens it"
    ),
    "constriction_by_branch_order": (
        "Factors replacing \"Pericyte constriction/dilation factor\" for the branch orders "
        "listed, e.g. {B01: 1.0}. Orders not listed, or an empty map, use that factor"
    ),
    "run_perturbations": (
        "Re-solve the finished network once for each perturbation in the list, each starting "
        "from the same baseline"
    ),
    "perturbations": (
        "The perturbations to run, each a name, a type and that type's options. Each re-solves "
        "the baseline network, not the result of the one before"
    ),
    "perturbation_output_dir": (
        "Folder for the perturbations' results, one subfolder per perturbation named after it "
        "and its type. Empty uses the run's output folder"
    ),
    "run_pericyte_dilation_sweep": (
        "Sweep pericyte constriction/dilation against inlet pressure after the run. Read only by"
        " the whole-brain example script; in the panel, add a pericyte constriction/dilation "
        "sweep perturbation"
    ),
    "pericyte_dilation_min_percent": (
        "Start of the pericyte sweep: every vessel's diameter changed by this percentage, with "
        "the pericyte sites on top. Negative constricts, positive dilates"
    ),
    "pericyte_dilation_max_percent": (
        "End of the pericyte sweep: every vessel's diameter changed by this percentage, with the"
        " pericyte sites on top. Negative constricts, positive dilates"
    ),
    "pericyte_dilation_step_percent": (
        "Step between diameter changes in the pericyte sweep. Always positive, though the range "
        "may run negative"
    ),
    "arteriole_diameter_change_percent": (
        "Change every arteriole's diameter along its whole length by this percentage: positive "
        "dilates, negative is a constriction (-20 narrows by 20%)"
    ),
    "capillary_block_selection": (
        "How the vessels to block are chosen: branch_order_probability (default) blocks a random"
        " share of chosen branch orders; vessel_ids blocks the vessels you list"
    ),
    "capillary_block_branch_orders": (
        "Branch orders whose vessels may be blocked, as a list such as ['BO2', 'BO3']. A bare "
        "number means that capillary order (2 is BO2)"
    ),
    "capillary_block_probability": (
        "Share of those branch orders' vessels to block, picked at random. Always exactly that "
        "share: 0.1 of 50 vessels blocks 5"
    ),
    "capillary_block_seed": (
        "Random seed that picks the blocked vessels, so a run repeats exactly; another seed "
        "blocks a different set"
    ),
    "capillary_block_replicates": (
        "Number of random draws of the block, each from the next seed. Above 1, each result's "
        "mean, spread and 95% range over the draws is reported too"
    ),
    "capillary_block_vessel_ids": (
        "Vessels to block, by the branchID shown when hovering over a vessel after the flow "
        "solve, e.g. [12, 40]"
    ),
    "capillary_block_mask_path": (
        "Mask of the blocks, on the segmented image's grid: each separate blob is one block and "
        "stops the vessel it lies in"
    ),
    "capillary_block_mask_h5_dataset_name": "Name of the dataset holding the block mask inside an H5 file; ignored for TIFF masks",
    "capillary_block_mask_min_overlap_fraction": (
        "Share of a blob that must lie inside the segmented vessels for it to block anything; a "
        "blob mostly outside them blocks nothing"
    ),
    "capillary_block_mask_vessels_per_block": (
        "Which vessels a blob blocks: largest_overlap (default) the one holding most of it; "
        "every_overlapped_vessel every vessel it touches, as a blob at a junction may"
    ),
    "capillary_block_resistance_factor": (
        "Number a blocked vessel's resistance is multiplied by. Very large stops nearly all its "
        "flow while keeping the network solvable"
    ),
    "capillary_block_hypoperfusion_fraction": (
        "Count another vessel as hypoperfused when its flow falls by at least this fraction of "
        "its baseline flow"
    ),
    "arteriole_dilation_min_percent": (
        "Start of the arteriole sweep: every arteriole's diameter changed along its whole length"
        " by this percentage. Negative constricts, positive dilates"
    ),
    "arteriole_dilation_max_percent": (
        "End of the arteriole sweep: every arteriole's diameter changed along its whole length "
        "by this percentage. Negative constricts, positive dilates"
    ),
    "arteriole_dilation_step_percent": (
        "Step between diameter changes in the arteriole sweep. Always positive, though the range"
        " may run negative"
    ),
    "capillary_dilation_min_percent": (
        "Start of the capillary sweep: every capillary's diameter changed along its whole length"
        " by this percentage. Negative constricts, positive dilates"
    ),
    "capillary_dilation_max_percent": (
        "End of the capillary sweep: every capillary's diameter changed along its whole length "
        "by this percentage. Negative constricts, positive dilates"
    ),
    "capillary_dilation_step_percent": (
        "Step between diameter changes in the capillary sweep. Always positive, though the range"
        " may run negative"
    ),
    "pericyte_geometry_dilation_percent": (
        "Diameter change applied to every vessel throughout a pericyte spacing or length sweep, "
        "with the pericyte sites on top. Negative constricts, positive dilates"
    ),
    "constriction_spacing_min_um": "Pericyte spacing the spacing sweep starts at, centre to centre along each capillary",
    "constriction_spacing_max_um": "Pericyte spacing the spacing sweep ends at, centre to centre along each capillary",
    "constriction_spacing_step_um": "Step between pericyte spacings in the spacing sweep",
    "constriction_length_min_um": "Pericyte site length the length sweep starts at, measured along the vessel",
    "constriction_length_max_um": "Pericyte site length the length sweep ends at, measured along the vessel",
    "constriction_length_step_um": "Step between pericyte site lengths in the length sweep",
    "inlet_pressure_min_pa": "Inlet pressure the pressure sweep starts at",
    "inlet_pressure_max_pa": "Inlet pressure the pressure sweep ends at",
    "inlet_pressure_step_pa": "Step between inlet pressures in the pressure sweep",
    "sweep_output_dir": (
        "Folder for the whole-brain example script's sweep table and curves. A pericyte sweep "
        "perturbation writes into its own folder instead"
    ),
    # 9. Additional measurements
    "measurement_3d_to_cell_mask": (
        "Measure how far each object in a 3D mask (such as cells) lies from the nearest vessel, "
        "0 where they touch, and which vessel that is"
    ),
    "cell_mask_path": (
        "Binary 3D mask of the objects (such as cells) to measure, the same shape as the input "
        "image; each connected piece is one object"
    ),
    "cell_mask_h5_dataset_name": "Dataset to read from the cell mask when it is an H5 file",
    "measurement_3d_vessel_mask_path": "Vessel mask to measure the distances to. Leave empty to use the run's segmented image",
    "measurement_3d_vessel_mask_h5_dataset_name": "Dataset to read from that vessel mask when it is an H5 file",
    "statistics": "Report the network's vessel statistics, written to a CSV and shown in the Statistics tab",
    "statistics_basic": "Report node and vessel counts, total length, mean diameter and the other basic measures",
    "statistics_tortuosity": "Report how much each vessel's centreline deviates from a straight line",
    "statistics_branching": "Report how many branch points (junctions of three or more vessels) the network has",
    "statistics_tree_asymmetry": (
        "Report how unevenly the network branches: how different in size the sub-trees beyond "
        "each branch point are"
    ),
    "statistics_fractal_dimension": (
        "Report the network's fractal dimension, a measure of how it fills space, from its "
        "junction positions and from its centrelines"
    ),
    "statistics_vessel_density": (
        "Report vessel length and volume per unit of tissue volume: the measured tissue when "
        "that is on, otherwise the box the network spans"
    ),
    "statistics_vessel_classes": (
        "Report the same measures for each vessel class (arterioles, capillaries, venules, or "
        "classes by size): count, length, density, diameter, tortuosity and, once solved, flow "
        "and pressure"
    ),
    "statistics_murray_law": (
        "Report how closely vessel widths at each junction follow Murray's law: the parent's "
        "diameter cubed equals the sum of its daughters' cubed"
    ),
    "statistics_daughter_angles": "Report the angle between sibling daughter vessels at each branch point",
    "statistics_intercapillary_distance": (
        "Report the typical distance between neighbouring vessels, over all vessels and, once "
        "flow is solved, over those carrying flow"
    ),
    "statistics_junction_shape": (
        "Report the angles between the three vessels at each junction: smallest, largest, how "
        "often two lie within 45 degrees, and how flat junctions are"
    ),
    "statistics_tissue_coverage": (
        "Report how far tissue lies from the nearest vessel: the share of the image box within "
        "20 um and beyond 50 um, and the median and 90th percentile distance"
    ),
    "statistics_vessel_columns": (
        "Report vessels running along the stack (z) axis: their share of total length, how many "
        "per square mm, and how evenly they spread across the image"
    ),
    "statistics_dead_ends": (
        "Report dead ends by kind (leaving the image, broken, blind ...), how many there are per"
        " mm of vessel, and their lengths"
    ),
    "statistics_flow_face_margin_um": (
        "Leave vessels this close to the image edge out of the flow statistics, where flows are "
        "least certain. 0 keeps every vessel"
    ),
    "measure_tissue_volume": (
        "Measure the tissue's volume and surface from the raw image, so vessel density is per "
        "tissue volume rather than per box the network spans"
    ),
    "tissue_raw_tiff_path": (
        "Raw image to find the tissue in, on the segmentation's grid. Leave empty to use the "
        "run's raw image"
    ),
    "tissue_raw_channel": (
        "Channel to find the tissue in: one where the tissue's own background shows. Leave unset"
        " for a single-channel file, or to use the run's raw image channel"
    ),
    "tissue_threshold_method": (
        "How tissue is told from empty space in the raw image. otsu (default), li and triangle "
        "pick a brightness level automatically; manual uses your own"
    ),
    "tissue_manual_threshold": "Raw image brightness the tissue surface passes through: tissue above it, empty space below",
    "tissue_smoothing_sigma_um": (
        "How much to blur the raw image before finding the tissue edge, so noise and texture do "
        "not roughen it. Higher gives a smoother, rounder surface"
    ),
    "tissue_working_voxel_size_um": (
        "Block size the raw image is averaged into before finding the tissue. Larger is faster "
        "and uses less memory on big stacks"
    ),
    "tissue_min_component_fraction": (
        "Drop separate pieces of tissue smaller than this fraction of the largest piece, such as"
        " specks of debris"
    ),
    "tissue_fill_holes": "Count dark cavities enclosed by tissue on every side as tissue",
    "measure_pericytes": (
        "Measure around each pericyte in a pericyte mask, placed on its nearest capillary: the "
        "vessel's diameter along it and its distance to the nearest junction"
    ),
    "pericyte_measurement_mask_path": (
        "Mask of the pericytes, binary or labelled, the same shape as the input image; each "
        "connected piece is one pericyte"
    ),
    "pericyte_measurement_mask_h5_dataset_name": "Dataset to read from the pericyte mask when it is an H5 file",
    "measure_pericyte_diameter_profiles": (
        "Record the vessel's diameter in steps either side of each pericyte, a CSV per pericyte,"
        " and each branch's resistance with and without the measured narrowing"
    ),
    "pericyte_profile_step_um": (
        "Spacing of the diameter readings along the vessel from each pericyte. Smaller gives a "
        "finer profile"
    ),
    "pericyte_profile_reach_um": (
        "How far along the vessel each way from a pericyte its diameter is read, stopping sooner"
        " where the vessel ends"
    ),
    "pericyte_profile_reading": (
        "Where the profile's diameters come from: segmentation (default), the mask's "
        "cross-section at each point; or raw_image_fwhm, a width fitted across the vessel in the"
        " raw image"
    ),
    "measure_pericyte_junction_distances": (
        "Measure how far each pericyte is from its vessel's nearest junction, along the vessel "
        "and in a straight line"
    ),
    "pericyte_measurement_max_distance_um": (
        "Measure a pericyte only when its centre lies within this distance of the nearest "
        "capillary's centreline. Higher takes in pericytes further from a vessel"
    ),
    "pericyte_measurement_min_diameter_um": (
        "Leave out mask pieces narrower than this (the width of a sphere of the same volume) as "
        "too small to be a pericyte"
    ),
    "pericyte_measurement_max_diameter_um": (
        "Leave out mask pieces wider than this (the width of a sphere of the same volume) as too"
        " large to be a pericyte"
    ),
    "statistics_network_analysis": (
        "Run the connectivity and network analyses: weak points, loops, inlet-to-outlet routes, "
        "centrality and communities"
    ),
    "statistics_mode": (
        "How exactly the slower network analyses are computed. fast (default) samples or caps "
        "them; full computes them exactly, which can take far longer on a large network"
    ),
    "statistics_network_robustness": (
        "Report how much of the network hangs on single vessels or junctions whose loss would "
        "split it and, with inlets and outlets, which would cut an inlet off from an outlet"
    ),
    "statistics_bottlenecks": (
        "Report the smallest set of vessels whose blockage would cut every inlet off from every "
        "outlet, and each vessel's share of the shortest inlet-to-outlet routes"
    ),
    "statistics_shunts": (
        "Report shunts: vessels on inlet-to-outlet routes much shorter than the typical route, "
        "which bypass most of the capillary bed"
    ),
    "statistics_route_weighting": (
        "What routes are measured in for the route-based analyses: vessel count (topology), "
        "centreline length (default) or resistance, which falls back to length without "
        "haemodynamics"
    ),
    "statistics_shunt_max_route_fraction": (
        "Flag a vessel as a shunt when its shortest inlet-to-outlet route is at most this "
        "fraction of the median route. Lower flags fewer"
    ),
    "statistics_occlusion_impact": (
        "Report, for every vessel, the flow lost and the length of other vessels starved if it "
        "alone is blocked. Slow; fast mode tests only the 500 busiest vessels"
    ),
    "statistics_occlusion_hypoperfusion_fraction": (
        "Count another vessel as starved when its flow falls by at least this fraction after a "
        "blockage. Higher counts fewer"
    ),
    "statistics_occlusion_curves": (
        "Report how flow and inlet-outlet connection decline as more and more vessels are "
        "blocked (at random, busiest first, narrowest first). Slower"
    ),
    "statistics_occlusion_curve_max_fraction": "Keep blocking vessels until this fraction of them is blocked",
    "statistics_current_flow": (
        "Report each vessel's share of the inlet-to-outlet flow as it splits over parallel "
        "routes like current in a circuit, and the network's inlet-outlet resistance"
    ),
    "statistics_perfusion_territories": (
        "Report which inlet's and which outlet's territory each vessel lies in, and flag the "
        "watershed vessels on the borders between territories"
    ),
    "statistics_capillary_bed_joins": (
        "Report the capillary branch order at which each feeding arteriole's capillary bed first"
        " meets another's. Needs capillary branch orders"
    ),
    "statistics_capillary_path_orders": (
        "Report how many capillaries lie between arterioles and venules: for each capillary "
        "draining into a venule, the fewest capillaries back to an arteriole, and that route's "
        "length and straightness"
    ),
    "statistics_transit_time": (
        "Report how long blood takes to cross each vessel and to go from inlets to outlets, "
        "including how unevenly it crosses the capillaries"
    ),
    "statistics_flux_heterogeneity": (
        "Report how unevenly flow and velocity are spread over the vessels carrying flow, how "
        "much of the network carries flow, and predicted red-cell velocity, flux and spacing"
    ),
    "statistics_loop_hierarchy": (
        "Report the shortest loop through each vessel and the spread of loop lengths: the scale "
        "at which the network offers alternative routes"
    ),
    "statistics_strahler": (
        "Report Strahler orders of the arterial and venous trees and how vessel count, length "
        "and width change from one order to the next"
    ),
    "statistics_algebraic_connectivity": (
        "Report the network's algebraic connectivity, how hard it is to cut in two, and the "
        "natural two-way split it suggests"
    ),
    "statistics_cyclomatic_number": (
        "Report how many independent loops the network has. 0 means a pure tree, where every "
        "vessel is the only route to what lies beyond it"
    ),
    "statistics_degree_assortativity": (
        "Report whether junctions joining many vessels tend to link to other such junctions "
        "(vascular trees usually do not)"
    ),
    "statistics_rich_club": (
        "Report whether the best-connected junctions link to each other, forming a backbone of "
        "their own"
    ),
    "statistics_k_core": (
        "Report how far the network has a densely joined, redundant core inside a shell, rather "
        "than being purely tree-like"
    ),
    "statistics_flow_hierarchy": (
        "Report the share of vessels whose solved flow never runs round a loop. Below 1 points "
        "to a modelling problem, not anatomy"
    ),
    "statistics_path_efficiency": (
        "Report how close the shortest routes through the network come to straight-line "
        "distances, sampled in fast mode and exact in full"
    ),
    "statistics_community": (
        "Report the network's communities, groups of junctions more joined to each other than to"
        " the rest; large networks get a cruder split unless \"Statistics mode\" is full"
    ),
    "statistics_betweenness": (
        "Report betweenness centrality: how many shortest routes through the network pass "
        "through each junction, approximate on large networks unless \"Statistics mode\" is full"
    ),
    "compute_vascular_communities": (
        "Split the network into vascular communities, groups of vessels more joined to each "
        "other than to the rest, and offer them as a vessel colour option"
    ),
    "vascular_community_weighting": (
        "What the community split weighs vessels by: topology (default, connections only), "
        "length, resistance or flow. Resistance and flow need a solved network"
    ),
    # 10. Export
    "base_plot_dir": "Folder the plot files (PNG images and HTML pages) are written under",
    "visualize_results": (
        "Write plot files: PNG images and interactive HTML pages, including a 3D view of the "
        "final network. Off skips them all and saves time"
    ),
    "interactive_plots": "Show the plots in interactive windows instead of saving them to files",
    "show_plots_in_ide": "Also open the saved plots in windows as the run goes on",
    "ide_plot_mode": (
        "Which saved plots open in windows. final_only (default) opens just the final network "
        "view; all adds the degree and branch-order plots; none opens none"
    ),
    "hold_ide_plots_open": "Wait at the end of the run until the plot windows are closed, so they stay open",
    "final_render_mode": (
        "Draw the final network plot as a flat 2d overlay image or as an interactive 3d page "
        "(default)"
    ),
    "vtk_export": (
        "Save the vessels, nodes and pericytes as VTK files, which 3D viewers such as ParaView "
        "can open"
    ),
    "visualize_vtk": "Open the exported VTK files in a separate 3D viewer window when the run finishes",
    "flow_direction_colouring": "Offer the flow direction's z, y and x parts as colour options for the flow arrows layer",
    "flow_arrow_scale": (
        "Starting size of the flow arrows in the viewer. Higher draws longer arrows; the layer's"
        " length control changes it later"
    ),
    "verbose_logging": "Write more detailed messages to the log as each stage runs",
    "export_citations": "Write a text file citing the software and models this run actually used",
    "vtk_output_prefix": (
        "Where the run's outputs are written: its folder holds the output files, and the VTK "
        "files are named after its last part"
    ),
}


def tooltip_for(setting: Setting) -> str:
    """The hover text for *setting*'s row, before the panel adds its unit."""
    return SETTING_TOOLTIPS.get(setting.name, setting.help)
