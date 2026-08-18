"""Utilities for running ilastik segmentation in headless mode."""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_ilastik_headless_segmentation(
    input_image_path: str | Path,
    classifier_path: str | Path,
    output_path: str | Path,
    ilastik_executable: str | Path = "ilastik.exe",
) -> Path:
    """Run ilastik headless segmentation for a single image.

    Parameters
    ----------
    input_image_path:
        Path to the image file to segment.
    classifier_path:
        Path to the ilastik project/classifier file (.ilp).
    output_path:
        Path to write the segmented output image.
    ilastik_executable:
        ilastik executable. Defaults to ``ilastik.exe`` and can also be a full path.
    """
    input_image_path = Path(input_image_path)
    classifier_path = Path(classifier_path)
    output_path = Path(output_path)

    if not input_image_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier/project file not found: {classifier_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".tif", ".tiff", ".h5"}:
        raise ValueError(
            "Unsupported ilastik output extension. "
            "Use one of: .tif, .tiff, .h5"
        )

    # ilastik expects output placeholders in the pattern string.
    output_pattern = str(output_path.parent / f"{{nickname}}{output_suffix}")
    output_format = "hdf5" if output_suffix == ".h5" else "multipage tiff"
    command = [
        str(ilastik_executable),
        "--headless",
        f"--project={classifier_path}",
        "--export_source=Simple Segmentation",
        f"--output_format={output_format}",
        f"--output_filename_format={output_pattern}",
        str(input_image_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find ilastik executable '{ilastik_executable}'. "
            "Set ILASTIK_EXECUTABLE to the full ilastik path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ilastik segmentation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    generated_path = output_path.parent / f"{input_image_path.stem}{output_suffix}"
    if generated_path.exists():
        if generated_path != output_path:
            generated_path.replace(output_path)
    if not output_path.exists():
        raise RuntimeError(
            "ilastik finished, but no output file was found at "
            f"'{output_path}'. Expected generated path: '{generated_path}'."
        )
    return output_path
