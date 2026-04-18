import re
from pathlib import Path

content = Path("examples/carotid_image_to_model.py").read_text()

start_marker = "# Ilastik configuration settings"
end_marker = '"""Configuration defaults for diameter maps."""'

clean_globals = """# Ilastik configuration settings
RUN_ILASTIK = False
ILASTIK_OUTPUT_PROBABILITIES = True # Set to True for Probabilities, False for Simple Segmentation
ILASTIK_BINARY_PATH = "/home/dsas627/Desktop/ilastik-1.4.1rc2-gpu-Linux/run_ilastik.sh"
ILASTIK_PROJECT_PATH = root_dir / "examples" / "images" / "cb_wky_2x2x2_A.ilp"
RAW_IMAGE_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_input_images"
ILASTIK_OUTPUT_DIR = root_dir / "examples" / "images" / "ilastik_batch_processing_output_images"

# Paths for multi-input Ilastik features (e.g., Raw + Frangi)
RAW_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vessels.tif"
FRANGI_IMAGE_PATH = RAW_IMAGE_DIR / "C1-CB3-WKY-CB-A-2x2x2_vesselness_map.tif"

INPUT_PATH = None
H5_DATASET_NAME = None  # For h5 input, e.g. "data"

"""

start_pos = content.find(start_marker)
end_pos = content.find(end_marker)

if start_pos != -1 and end_pos != -1:
    content = content[:start_pos] + clean_globals + content[end_pos:]

old_pipeline_config = """@dataclass
class PipelineConfig:
    do_skeletonize: bool = True
    do_graph_building: bool = True
    do_resistance_calculation: bool = True
    verbose_logging: bool = False
    min_branch_length: int = 10
    vtk_output_prefix: Path = None
    plot_dir: Path = None"""

new_pipeline_config = """@dataclass
class PipelineConfig:
    do_skeletonize: bool = True
    do_graph_building: bool = True
    do_resistance_calculation: bool = True
    verbose_logging: bool = False
    min_branch_length: int = 10
    vtk_output_prefix: Path = Path(__file__).resolve().parents[1] / "examples" / "outputs" / "resistance_network"
    plot_dir: Path = Path(__file__).resolve().parents[1] / "examples" / "plots" / "carotid" """

content = content.replace(old_pipeline_config, new_pipeline_config)

bottom_call_regex = re.compile(r'    pre_config = PreprocessingConfig\(.*?\n    \)', re.DOTALL)
new_instantiation = """    # Pipeline Configurations are now fully self-contained in their dataclasses at the top of the file.
    pre_config = PreprocessingConfig()
    skel_config = SkeletonConfig()
    graph_config = GraphConfig()
    hemo_config = HaemodynamicsConfig(diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER, constrict_at_pericytes=False)
    vis_config = VisualizationConfig()
    pipeline_config = PipelineConfig()"""

content = bottom_call_regex.sub(new_instantiation, content)

main_block_old = 'if __name__ == "__main__":\n    plot_dir = BASE_PLOT_DIR / "carotid"\n    \n    # 1. Run Ilastik'
main_block_new = 'if __name__ == "__main__":\n    # 1. Run Ilastik'
content = content.replace(main_block_old, main_block_new)

# Wait, BASE_PLOT_DIR is also used above, wait, I removed BASE_PLOT_DIR from globals, let me check if it's used elsewhere.
# Actually I removed BASE_PLOT_DIR definition, so `plot_dir = BASE_PLOT_DIR / "carotid"` would crash if I didn't remove it.

Path("examples/carotid_image_to_model.py").write_text(content)
