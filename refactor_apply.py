import re
from pathlib import Path

content = Path("examples/carotid_image_to_model.py").read_text()

# 1. Add dataclasses after imports
imports = "import networkx as nx\nfrom dataclasses import dataclass, field"
content = content.replace("import networkx as nx", imports, 1)

dataclasses_code = """

@dataclass
class PreprocessingConfig:
    median_filter_size: int = 7
    probability_smoothing_sigma: float = 0.0
    morphological_opening_radius: int = 1
    enable_hysteresis_threshold: bool = True
    hysteresis_threshold_low: float = 0.2
    hysteresis_threshold_high: float = 0.4
    enable_hole_filling: bool = True
    ilastik_vessel_channel: int = 0
    enable_shannon_entropy: bool = True
    shannon_entropy_threshold: float = 0.95

@dataclass
class SkeletonConfig:
    closing_radius: int = 1
    bridge_gap_size: int = 1
    min_branch_length: int = 3
    max_bridge_distance: int = 0
    component_connectivity: int = 3
    min_component_percent: float = 5.0
    downsample_factor: float = 1.0
    use_padded_slicing: bool = True
    padded_slicing_padding: int = 3
    prune_mask_before: int = 1
    sub_volume_percentage: float = 0.25
    sub_volume_offset_z: float = 0.0
    sub_volume_offset_y: float = 0.0
    sub_volume_offset_x: float = 0.0
    bundle_scan_size: int = 9
    bundle_density_fraction: float = 0.025
    bundle_max_connections: int = 5
    bundle_hub_min_spacing: int = 0

@dataclass
class GraphConfig:
    keep_largest_component_only: bool = True
    edge_percent: float = 25.0
    end_percent: float = 25.0
    node_edge_axis: int = 0
    starting_nodes: list = field(default_factory=list)
    output_nodes: list = field(default_factory=list)

@dataclass
class HaemodynamicsConfig:
    constrict_at_pericytes: bool = False
    input_p_bc: float = 1000.0
    output_p_bc: float = 500.0
    diameter_by_branch_order: dict = field(default_factory=dict)

@dataclass
class VisualizationConfig:
    visualize_results: bool = False
    visualize_mask_only: bool = False
    visualize_vedo: bool = True
    visualize_overlay_preview: bool = False
    visualize_vedo_mode: str = 'iso'
    visualize_vedo_smooth_iter: int = 15
    visualize_vedo_spacing: tuple = (1.0, 1.0, 1.0)
    visualize_vedo_auto_spacing: bool = True
    visualize_vedo_opacity: float = 0.5
    visualize_mask_opacity: float = 1.0
    visualize_vtk: bool = False
    visualize_post_processed_mask: bool = False

@dataclass
class PipelineConfig:
    do_skeletonize: bool = True
    do_graph_building: bool = True
    do_resistance_calculation: bool = True
    verbose_logging: bool = False
    min_branch_length: int = 10
    vtk_output_prefix: Path = None
    plot_dir: Path = None
"""

content = content.replace("class IlastikClassifier():", dataclasses_code + "\nclass IlastikClassifier():", 1)

new_signature = """def carotid_image_to_model(image_path: Path | str, 
                           pre_config: PreprocessingConfig = None,
                           skel_config: SkeletonConfig = None,
                           graph_config: GraphConfig = None,
                           hemo_config: HaemodynamicsConfig = None,
                           vis_config: VisualizationConfig = None,
                           pipeline_config: PipelineConfig = None) -> None:
    if pre_config is None: pre_config = PreprocessingConfig()
    if skel_config is None: skel_config = SkeletonConfig()
    if graph_config is None: graph_config = GraphConfig()
    if hemo_config is None: hemo_config = HaemodynamicsConfig()
    if vis_config is None: vis_config = VisualizationConfig()
    if pipeline_config is None: pipeline_config = PipelineConfig()

    # Unpack configurations for backward compatibility within the function body
    diameter_by_branch_order = hemo_config.diameter_by_branch_order
    plot_dir = pipeline_config.plot_dir if pipeline_config.plot_dir is not None else Path("plots")
    verbose_logging = pipeline_config.verbose_logging
    do_skeletonize = pipeline_config.do_skeletonize
    do_graph_building = pipeline_config.do_graph_building
    do_resistance_calculation = pipeline_config.do_resistance_calculation
    constrict_at_pericytes = hemo_config.constrict_at_pericytes
    min_branch_length = pipeline_config.min_branch_length
    vtk_output_prefix = pipeline_config.vtk_output_prefix if pipeline_config.vtk_output_prefix is not None else Path("outputs/resistance_network")
    skeleton_closing_radius = skel_config.closing_radius
    skeleton_bridge_gap_size = skel_config.bridge_gap_size
    skeleton_min_branch_length = skel_config.min_branch_length
    skeleton_max_bridge_distance = skel_config.max_bridge_distance
    skeleton_component_connectivity = skel_config.component_connectivity
    skeleton_min_component_percent = skel_config.min_component_percent
    skeleton_downsample_factor = skel_config.downsample_factor
    skeleton_use_padded_slicing = skel_config.use_padded_slicing
    skeleton_padded_slicing_padding = skel_config.padded_slicing_padding
    skeleton_prune_mask_before = skel_config.prune_mask_before
    skeleton_sub_volume_percentage = skel_config.sub_volume_percentage
    skeleton_sub_volume_offset_z = skel_config.sub_volume_offset_z
    skeleton_sub_volume_offset_y = skel_config.sub_volume_offset_y
    skeleton_sub_volume_offset_x = skel_config.sub_volume_offset_x
    edge_percent = graph_config.edge_percent
    end_percent = graph_config.end_percent
    node_edge_axis = graph_config.node_edge_axis
    starting_nodes = graph_config.starting_nodes
    output_nodes = graph_config.output_nodes
    input_p_bc = hemo_config.input_p_bc
    output_p_bc = hemo_config.output_p_bc
    visualize_results = vis_config.visualize_results
    visualize_mask_only = vis_config.visualize_mask_only
    visualize_vedo = vis_config.visualize_vedo
    visualize_overlay_preview = vis_config.visualize_overlay_preview
    visualize_vedo_mode = vis_config.visualize_vedo_mode
    visualize_vedo_smooth_iter = vis_config.visualize_vedo_smooth_iter
    visualize_vedo_spacing = vis_config.visualize_vedo_spacing
    visualize_vedo_auto_spacing = vis_config.visualize_vedo_auto_spacing
    visualize_vedo_opacity = vis_config.visualize_vedo_opacity
    visualize_mask_opacity = vis_config.visualize_mask_opacity
    visualize_vtk = vis_config.visualize_vtk
    median_filter_size = pre_config.median_filter_size
    probability_smoothing_sigma = pre_config.probability_smoothing_sigma
    morphological_opening_radius = pre_config.morphological_opening_radius
    enable_hysteresis_threshold = pre_config.enable_hysteresis_threshold
    hysteresis_threshold_low = pre_config.hysteresis_threshold_low
    hysteresis_threshold_high = pre_config.hysteresis_threshold_high
    enable_hole_filling = pre_config.enable_hole_filling
    visualize_post_processed_mask = vis_config.visualize_post_processed_mask
    ilastik_vessel_channel = pre_config.ilastik_vessel_channel
    enable_shannon_entropy = pre_config.enable_shannon_entropy
    shannon_entropy_threshold = pre_config.shannon_entropy_threshold
    graph_keep_largest_component_only = graph_config.keep_largest_component_only
    bundle_scan_size = skel_config.bundle_scan_size
    bundle_density_fraction = skel_config.bundle_density_fraction
    bundle_max_connections = skel_config.bundle_max_connections
    bundle_hub_min_spacing = skel_config.bundle_hub_min_spacing"""

# Replace the old signature
import re
signature_regex = re.compile(r'def carotid_image_to_model\(.*?\) -> None:', re.DOTALL)
content = signature_regex.sub(new_signature, content)


# Now for the call at the bottom
call_regex = re.compile(r'    carotid_image_to_model\(\s*image_path=target_input_mask_path,.*?bundle_hub_min_spacing=SKELETON_BUNDLE_HUB_MIN_SPACING\n    \)', re.DOTALL)

new_call = """    pre_config = PreprocessingConfig(
        median_filter_size=MEDIAN_FILTER_SIZE,
        probability_smoothing_sigma=PROBABILITY_SMOOTHING_SIGMA,
        morphological_opening_radius=MORPHOLOGICAL_OPENING_RADIUS,
        enable_hysteresis_threshold=ENABLE_HYSTERESIS_THRESHOLD,
        hysteresis_threshold_low=HYSTERESIS_THRESHOLD_LOW,
        hysteresis_threshold_high=HYSTERESIS_THRESHOLD_HIGH,
        enable_hole_filling=ENABLE_HOLE_FILLING,
        ilastik_vessel_channel=ILASTIK_VESSEL_CHANNEL,
        enable_shannon_entropy=ENABLE_SHANNON_ENTROPY,
        shannon_entropy_threshold=SHANNON_ENTROPY_THRESHOLD,
    )
    skel_config = SkeletonConfig(
        closing_radius=SKELETON_CLOSING_RADIUS,
        bridge_gap_size=SKELETON_BRIDGE_GAP_SIZE,
        min_branch_length=SKELETON_MIN_BRANCH_LENGTH,
        max_bridge_distance=SKELETON_MAX_BRIDGE_DISTANCE,
        component_connectivity=SKELETON_COMPONENT_CONNECTIVITY,
        min_component_percent=SKELETON_MIN_COMPONENT_PERCENT,
        downsample_factor=SKELETON_DOWNSAMPLE_FACTOR,
        use_padded_slicing=SKELETON_USE_PADDED_SLICING,
        padded_slicing_padding=SKELETON_PADDED_SLICING_PADDING,
        prune_mask_before=SKELETON_PRUNE_MASK_BEFORE_SKELETONIZATION,
        sub_volume_percentage=SKELETON_SUB_VOLUME_PERCENTAGE,
        sub_volume_offset_z=SKELETON_SUB_VOLUME_CENTER_OFFSET_Z,
        sub_volume_offset_y=SKELETON_SUB_VOLUME_CENTER_OFFSET_Y,
        sub_volume_offset_x=SKELETON_SUB_VOLUME_CENTER_OFFSET_X,
        bundle_scan_size=SKELETON_BUNDLE_SCAN_SIZE,
        bundle_density_fraction=SKELETON_BUNDLE_DENSITY_FRACTION,
        bundle_max_connections=SKELETON_BUNDLE_MAX_CONNECTIONS,
        bundle_hub_min_spacing=SKELETON_BUNDLE_HUB_MIN_SPACING,
    )
    graph_config = GraphConfig(
        keep_largest_component_only=GRAPH_KEEP_LARGEST_COMPONENT_ONLY,
        edge_percent=EDGE_PERCENT,
        end_percent=END_PERCENT,
        node_edge_axis=NODE_EDGE_AXIS,
        starting_nodes=STARTING_NODES,
        output_nodes=OUTPUT_NODES,
    )
    hemo_config = HaemodynamicsConfig(
        constrict_at_pericytes=CONSTRICT_AT_PERICYTES,
        input_p_bc=INPUT_P_BC,
        output_p_bc=OUTPUT_P_BC,
        diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
    )
    vis_config = VisualizationConfig(
        visualize_results=VISUALIZE_RESULTS,
        visualize_mask_only=VISUALIZE_MASK_ONLY,
        visualize_vedo=VISUALIZE_VEDO,
        visualize_overlay_preview=VISUALIZE_OVERLAY_PREVIEW,
        visualize_vedo_mode=VISUALIZE_VEDO_MODE,
        visualize_vedo_smooth_iter=VISUALIZE_VEDO_SMOOTH_ITER,
        visualize_vedo_spacing=VISUALIZE_VEDO_SPACING,
        visualize_vedo_auto_spacing=VISUALIZE_VEDO_AUTO_SPACING,
        visualize_vedo_opacity=VISUALIZE_VEDO_OPACITY,
        visualize_mask_opacity=VISUALIZE_MASK_OPACITY,
        visualize_vtk=VISUALIZE_VTK,
        visualize_post_processed_mask=VISUALIZE_POST_PROCESSED_MASK,
    )
    pipeline_config = PipelineConfig(
        do_skeletonize=DO_SKELETONIZE,
        do_graph_building=DO_GRAPH_BUILDING,
        do_resistance_calculation=DO_RESISTANCE_CALCULATION,
        verbose_logging=VERBOSE_LOGGING,
        min_branch_length=MIN_BRANCH_LENGTH,
        vtk_output_prefix=VTK_OUTPUT_PREFIX,
        plot_dir=plot_dir,
    )

    carotid_image_to_model(
        image_path=target_input_mask_path,
        pre_config=pre_config,
        skel_config=skel_config,
        graph_config=graph_config,
        hemo_config=hemo_config,
        vis_config=vis_config,
        pipeline_config=pipeline_config
    )"""

content = call_regex.sub(new_call, content)

Path("examples/carotid_image_to_model.py").write_text(content)
