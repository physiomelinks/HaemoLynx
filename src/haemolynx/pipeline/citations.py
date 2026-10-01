"""Which third-party packages and haemodynamics models a run actually used,
and how to cite each one.

The pipeline links many building blocks: some run unconditionally (NumPy,
NetworkX), others only when a setting turns them on (ilastik segmentation,
VTK export, the Pries viscosity law, the Pries-Secomb haematocrit
distribution, a pericyte-constriction perturbation). A citation list that
named every dependency the pipeline can ever touch, regardless of whether
this run used it, would misattribute methods nobody ran. This module decides
-- from *settings* alone, the same values that drove the run, plus whether
napari is the interpreter running it -- exactly which of them applied, and
renders the result as one plain-text reading list.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .checks import _ILASTIK_VESSEL_MASK_PREREQUISITES

#: HaemoLynx's own citation, always first: the pipeline that ran everything
#: else below, not one more optional dependency.
SOFTWARE_CITATION = (
    "HaemoLynx (Argus, F., Davis, H.). "
    "https://github.com/physiomelinks/HaemoLynx"
)


@dataclass(frozen=True)
class Citation:
    """One thing to cite: what it is, why it's relevant, and the reference text."""

    name: str
    reason: str
    reference: str
    #: True when *settings* show this run actually used it.
    applies: Callable[[Mapping[str, Any]], bool]


def _used_ilastik(settings: Mapping[str, Any]) -> bool:
    """Whether ilastik actually ran for this settings dict.

    The large/small-vessel-mask flags each need their own further settings on
    too -- see :data:`~haemolynx.pipeline.checks._ILASTIK_VESSEL_MASK_PREREQUISITES`,
    the one table both this function and ``check_ilastik_vessel_mask_prerequisites``
    read, so a flag left on with an unmet prerequisite (the "set but
    ineffective" case the schema itself already warns about) cannot credit
    ilastik here without also failing preflight.
    """
    if settings.get("use_ilastik_segmentation"):
        return True
    return any(
        settings.get(flag) and all(settings.get(name) for name in requires)
        for flag, requires in _ILASTIK_VESSEL_MASK_PREREQUISITES.items()
    )


def _input_suffix(settings: Mapping[str, Any]) -> str:
    path = settings.get("input_path")
    return "" if path is None else Path(path).suffix.lower()


def _used_hdf5_input(settings: Mapping[str, Any]) -> bool:
    return _input_suffix(settings) in {".h5", ".hdf5"}


def _used_tiff_input(settings: Mapping[str, Any]) -> bool:
    return _input_suffix(settings) not in {".h5", ".hdf5"}


def _used_plotly_3d(settings: Mapping[str, Any]) -> bool:
    return bool(
        settings.get("visualize_results")
        and settings.get("final_render_mode") == "3d"
    )


def _used_vtk_export(settings: Mapping[str, Any]) -> bool:
    return bool(settings.get("run_haemodynamics") and settings.get("vtk_export"))


def _used_pries_viscosity(settings: Mapping[str, Any]) -> bool:
    return bool(settings.get("run_haemodynamics")) and settings.get("viscosity_law") == "pries"


def _used_capillary_power_law(settings: Mapping[str, Any]) -> bool:
    return (
        bool(settings.get("run_haemodynamics"))
        and settings.get("viscosity_law") == "capillary_power_law"
    )


def _used_haematocrit_distribution(settings: Mapping[str, Any]) -> bool:
    return (
        bool(settings.get("run_haemodynamics"))
        and settings.get("haematocrit_model") == "distributed_iterative"
    )


def _used_secomb_junction_rule(settings: Mapping[str, Any]) -> bool:
    return _used_haematocrit_distribution(settings) and settings.get(
        "haematocrit_junction_rule", "no_separation"
    ) in ("no_separation", "sequential_bifurcations")


#: Perturbation types that place focal pericyte constrictions -- kept as a
#: literal tuple rather than importing haemodynamics.perturbations, so this
#: module (and the settings-only text it produces) stays a leaf the rest of
#: the pipeline schema can depend on without a cycle.
_PERICYTE_CONSTRICTION_PERTURBATION_TYPES = frozenset(
    {
        "pressure_and_pericyte_sweep",
        "pericyte_dilation_sweep",
        "pericyte_spacing_sweep",
        "pericyte_length_sweep",
        "pericyte_diameter_change",
        "arteriole_and_pericyte_diameter_change",
    }
)


def _used_pericyte_constriction(settings: Mapping[str, Any]) -> bool:
    if not settings.get("run_perturbations"):
        return False
    entries = settings.get("perturbations") or ()
    return any(
        str(entry.get("type")) in _PERICYTE_CONSTRICTION_PERTURBATION_TYPES
        for entry in entries
    )


def _used_napari(settings: Mapping[str, Any]) -> bool:
    # Not a setting: both a CLI run and a napari-panel run share the same
    # settings dict, so the only real signal that this call came from the
    # panel is that napari is already the interpreter running it.
    return "napari" in sys.modules


#: Always checked in this order; the rendered file lists whichever apply.
PACKAGE_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="NumPy",
        reason="array storage and numerics throughout the pipeline",
        reference=(
            "Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020). "
            "Array programming with NumPy. Nature, 585, 357-362. "
            "https://doi.org/10.1038/s41586-020-2649-2"
        ),
        applies=lambda settings: True,
    ),
    Citation(
        name="SciPy",
        reason="linear algebra and spatial routines",
        reference=(
            "Virtanen, P., Gommers, R., Oliphant, T.E. et al. (2020). SciPy "
            "1.0: fundamental algorithms for scientific computing in "
            "Python. Nature Methods, 17, 261-272. "
            "https://doi.org/10.1038/s41592-019-0686-2"
        ),
        applies=lambda settings: True,
    ),
    Citation(
        name="NetworkX",
        reason="the vessel graph representation",
        reference=(
            "Hagberg, A.A., Schult, D.A., Swart, P.J. (2008). Exploring "
            "network structure, dynamics, and function using NetworkX. "
            "Proceedings of the 7th Python in Science Conference "
            "(SciPy2008), 11-15."
        ),
        applies=lambda settings: True,
    ),
    Citation(
        name="scikit-image",
        reason="skeletonization of the segmented vessel mask",
        reference=(
            "van der Walt, S., Schonberger, J.L., Nunez-Iglesias, J. et al. "
            "(2014). scikit-image: image processing in Python. PeerJ, 2, "
            "e453. https://doi.org/10.7717/peerj.453"
        ),
        applies=lambda settings: bool(
            settings.get("do_skeletonize", True) or settings.get("measure_tissue_volume")
        ),
    ),
    Citation(
        name="skan",
        reason="skeleton-to-graph conversion",
        reference=(
            "Nunez-Iglesias, J., Blanch, A.J., Looker, O. et al. (2018). A "
            "new Python library to analyse skeleton images confirms "
            "malaria parasite remodelling of the red blood cell membrane "
            "skeleton. PeerJ, 6, e4312. https://doi.org/10.7717/peerj.4312"
        ),
        applies=lambda settings: bool(settings.get("do_graph_building", True)),
    ),
    Citation(
        name="pandas",
        reason="statistics tables and CSV export",
        reference=(
            "McKinney, W. (2010). Data structures for statistical "
            "computing in Python. Proceedings of the 9th Python in Science "
            "Conference, 56-61."
        ),
        applies=lambda settings: bool(settings.get("statistics")),
    ),
    Citation(
        name="Matplotlib",
        reason="the static overlay and degree-distribution plots",
        reference=(
            "Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. "
            "Computing in Science & Engineering, 9(3), 90-95. "
            "https://doi.org/10.1109/MCSE.2007.55"
        ),
        applies=lambda settings: bool(settings.get("visualize_results")),
    ),
    Citation(
        name="Plotly",
        reason="the interactive 3D graph views",
        reference=(
            "Plotly Technologies Inc. (2015). Collaborative data science. "
            "Plotly Technologies Inc., Montreal, QC. https://plot.ly"
        ),
        applies=_used_plotly_3d,
    ),
    Citation(
        name="PyVista / VTK",
        reason="the exported .vtp vessel/pericyte/node files",
        reference=(
            "Sullivan, C., Kaszynski, A. (2019). PyVista: 3D plotting and "
            "mesh analysis through a streamlined interface for the "
            "Visualization Toolkit (VTK). Journal of Open Source Software, "
            "4(37), 1450. https://doi.org/10.21105/joss.01450\n"
            "    Schroeder, W., Martin, K., Lorensen, B. (2006). The "
            "Visualization Toolkit (4th ed.). Kitware."
        ),
        applies=_used_vtk_export,
    ),
    Citation(
        name="h5py",
        reason="loading the HDF5 input image",
        reference="Collette, A. (2013). Python and HDF5. O'Reilly Media.",
        applies=_used_hdf5_input,
    ),
    Citation(
        name="tifffile",
        reason="loading the TIFF input image",
        reference=(
            "Gohlke, C. tifffile: read and write TIFF files with Python. "
            "https://github.com/cgohlke/tifffile"
        ),
        applies=_used_tiff_input,
    ),
    Citation(
        name="ilastik",
        reason="the headless pixel-classification segmentation step",
        reference=(
            "Berg, S., Kutra, D., Kroeger, T. et al. (2019). ilastik: "
            "interactive machine learning for (bio)image analysis. Nature "
            "Methods, 16, 1226-1232. "
            "https://doi.org/10.1038/s41592-019-0582-9"
        ),
        applies=_used_ilastik,
    ),
    Citation(
        name="napari",
        reason="the interactive viewer this run was driven from",
        reference=(
            "napari contributors (2019). napari: a multi-dimensional image "
            "viewer for Python. https://doi.org/10.5281/zenodo.3555620"
        ),
        applies=_used_napari,
    ),
)

MODEL_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="Hagen-Poiseuille law",
        reason="the base resistance model for every vessel segment",
        reference=(
            "Hagen, G. (1839). Uber die Bewegung des Wassers in engen "
            "zylindrischen Rohren. Poggendorffs Annalen der Physik und "
            "Chemie, 46, 423-442.\n"
            "    Poiseuille, J.L.M. (1840-1846). Recherches experimentales "
            "sur le mouvement des liquides dans les tubes de tres petits "
            "diametres. Comptes Rendus de l'Academie des Sciences."
        ),
        applies=lambda settings: bool(settings.get("run_haemodynamics")),
    ),
    Citation(
        name="Pries apparent-viscosity law (viscosity_law='pries')",
        reason="diameter- and haematocrit-dependent apparent blood viscosity",
        reference=(
            "Pries, A.R., Neuhaus, D., Gaehtgens, P. (1992). Blood "
            "viscosity in tube flow: dependence on diameter and "
            "hematocrit. American Journal of Physiology, 263(6 Pt 2), "
            "H1770-H1778.\n"
            "    Pries, A.R., Secomb, T.W., Gessner, T. et al. (1994). "
            "Resistance to blood flow in microvessels in vivo. "
            "Circulation Research, 75(5), 904-915.\n"
            "    Pries, A.R., Secomb, T.W. (2005). Microvascular blood "
            "viscosity in vivo and the endothelial surface layer. "
            "American Journal of Physiology - Heart and Circulatory "
            "Physiology, 289(6), H2657-H2664."
        ),
        applies=_used_pries_viscosity,
    ),
    Citation(
        name="Legacy capillary power-law viscosity (viscosity_law='capillary_power_law')",
        reason="the placeholder viscosity model kept for comparison with earlier results",
        reference=(
            "This pipeline's own one-point calibration against the Pries "
            "in vitro law at 5 um (see haemolynx.haemodynamics.viscosity); "
            "no specific published source is recorded for it."
        ),
        applies=_used_capillary_power_law,
    ),
    Citation(
        name="Pries-Secomb bifurcation haematocrit distribution (haematocrit_model='distributed_iterative')",
        reason="local discharge haematocrit from bifurcation phase separation",
        reference=(
            "Pries, A.R., Ley, K., Claassen, M., Gaehtgens, P. (1989). Red "
            "cell distribution at microvascular bifurcations. "
            "Microvascular Research, 38(1), 81-101.\n"
            "    Pries, A.R., Secomb, T.W. (2005). Microvascular blood "
            "viscosity in vivo and the endothelial surface layer. "
            "American Journal of Physiology - Heart and Circulatory "
            "Physiology, 289(6), H2657-H2664."
        ),
        applies=_used_haematocrit_distribution,
    ),
    Citation(
        name="Secomb's network-flow junction rules (haematocrit_junction_rule)",
        reason=(
            "how red cells divide at junctions the phase-separation law does "
            "not cover: no separation (dishem.cpp) or a chain of bifurcations "
            "(dishem_generalized.cpp)"
        ),
        reference=(
            "Secomb, T.W. NetFlowV2: simulation of microvascular network "
            "hemodynamics (dishem.cpp, dishem_generalized.cpp). "
            "https://github.com/secomb/NetFlowV2"
        ),
        applies=_used_secomb_junction_rule,
    ),
    Citation(
        name="Focal (pericyte) constriction model",
        reason="local vessel narrowing along a pericyte-typed perturbation",
        reference=(
            "This pipeline's own periodic-constriction model (see "
            "haemolynx.haemodynamics.constriction); not adapted from a "
            "single published source."
        ),
        applies=_used_pericyte_constriction,
    ),
)


def _used_tissue_threshold(method: str) -> Callable[[Mapping[str, Any]], bool]:
    def applies(settings: Mapping[str, Any]) -> bool:
        return bool(settings.get("measure_tissue_volume")) and (
            settings.get("tissue_threshold_method", "otsu") == method
        )

    return applies


#: Image-analysis methods outside the haemodynamics model: listed only when a
#: run used one, so a run without them renders exactly as it always has.
IMAGE_ANALYSIS_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="Tissue surface (measure_tissue_volume)",
        reason=(
            "the tissue's surface and volume: marching cubes through the raw "
            "image's background, smoothed by normalised convolution"
        ),
        reference=(
            "Lorensen, W.E., Cline, H.E. (1987). Marching cubes: a high "
            "resolution 3D surface construction algorithm. ACM SIGGRAPH "
            "Computer Graphics, 21(4), 163-169. "
            "https://doi.org/10.1145/37402.37422\n"
            "    Lewiner, T., Lopes, H., Vieira, A.W., Tavares, G. (2003). "
            "Efficient implementation of Marching Cubes' cases with "
            "topological guarantees. Journal of Graphics Tools, 8(2), 1-15. "
            "https://doi.org/10.1080/10867651.2003.10487582\n"
            "    Knutsson, H., Westin, C.-F. (1993). Normalized and "
            "differential convolution. Proceedings of IEEE Conference on "
            "Computer Vision and Pattern Recognition, 515-523. "
            "https://doi.org/10.1109/CVPR.1993.341081"
        ),
        applies=lambda settings: bool(settings.get("measure_tissue_volume")),
    ),
    Citation(
        name="Otsu threshold (tissue_threshold_method='otsu')",
        reason="splitting tissue from empty space in the raw image",
        reference=(
            "Otsu, N. (1979). A threshold selection method from gray-level "
            "histograms. IEEE Transactions on Systems, Man, and Cybernetics, "
            "9(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076"
        ),
        applies=_used_tissue_threshold("otsu"),
    ),
    Citation(
        name="Li minimum cross-entropy threshold (tissue_threshold_method='li')",
        reason="splitting tissue from empty space in the raw image",
        reference=(
            "Li, C.H., Lee, C.K. (1993). Minimum cross entropy thresholding. "
            "Pattern Recognition, 26(4), 617-625. "
            "https://doi.org/10.1016/0031-3203(93)90115-D\n"
            "    Li, C.H., Tam, P.K.S. (1998). An iterative algorithm for "
            "minimum cross entropy thresholding. Pattern Recognition Letters, "
            "19(8), 771-776. https://doi.org/10.1016/S0167-8655(98)00057-9"
        ),
        applies=_used_tissue_threshold("li"),
    ),
    Citation(
        name="Triangle threshold (tissue_threshold_method='triangle')",
        reason="splitting tissue from empty space in the raw image",
        reference=(
            "Zack, G.W., Rogers, W.E., Latt, S.A. (1977). Automatic "
            "measurement of sister chromatid exchange frequency. Journal of "
            "Histochemistry & Cytochemistry, 25(7), 741-753. "
            "https://doi.org/10.1177/25.7.70454"
        ),
        applies=_used_tissue_threshold("triangle"),
    ),
)


def used_citations(
    settings: Mapping[str, Any], citations: tuple[Citation, ...]
) -> list[Citation]:
    """The entries in *citations* whose ``applies`` holds for *settings*."""
    return [citation for citation in citations if citation.applies(settings)]


def _render_section(title: str, citations: list[Citation], empty_note: str) -> list[str]:
    lines = [title, "-" * len(title)]
    if not citations:
        lines.append(empty_note)
        lines.append("")
        return lines
    for citation in citations:
        lines.append(f"- {citation.name} ({citation.reason})")
        for reference_line in citation.reference.splitlines():
            lines.append(f"    {reference_line}")
        lines.append("")
    return lines


def render_citations(settings: Mapping[str, Any]) -> str:
    """The full citation-export text for one run, given its settings.

    Every entry's ``applies`` is checked against *settings* fresh -- nothing
    here is pulled in just because the pipeline can theoretically use it, only
    because this run's own configuration turned it on.
    """
    lines = [
        "HaemoLynx run citations",
        "========================",
        "",
        "Software and methods this run actually used, based on its own",
        "settings -- not a list of everything the pipeline can do.",
        "",
        "This pipeline:",
        f"    {SOFTWARE_CITATION}",
        "",
    ]
    lines.extend(
        _render_section(
            "Software packages",
            used_citations(settings, PACKAGE_CITATIONS),
            "(none detected)",
        )
    )
    lines.extend(
        _render_section(
            "Haemodynamics models and methods",
            used_citations(settings, MODEL_CITATIONS),
            "(none detected -- haemodynamics was not run)",
        )
    )
    image_analysis = used_citations(settings, IMAGE_ANALYSIS_CITATIONS)
    if image_analysis:
        lines.extend(_render_section("Image analysis methods", image_analysis, ""))
    return "\n".join(lines).rstrip() + "\n"


def write_citations(settings: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render and write the citation text file for one run, returning its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_citations(settings), encoding="utf-8")
    return output_path
