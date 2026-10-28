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

from haemolynx.graph.boundaries import vessel_classes_by_size, vessel_masks_in_use
from haemolynx.haemodynamics.perturbations import is_sweep_perturbation

from .checks import ilastik_segments_vessel_masks

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

    A large/small-vessel-mask flag counts only with its masks in use --
    :func:`~haemolynx.pipeline.checks.ilastik_segments_vessel_masks`, which
    preflight's ilastik check asks too -- so a flag left on under an unticked
    mask switch (hidden in the panel, and run as off) credits nothing.
    """
    if settings.get("use_ilastik_segmentation"):
        return True
    return ilastik_segments_vessel_masks(settings)


def _synthetic(settings: Mapping[str, Any]) -> bool:
    """A synthetic run (``network_source=synthetic``) reads no image, thins
    nothing and builds its graph from no skeleton."""
    from haemolynx.graph.boundaries import network_is_synthetic

    return network_is_synthetic(settings)


def _is_set(path: Any) -> bool:
    return path not in (None, "")


def _perturbation_entries(settings: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(type, settings it re-solves with)`` for every perturbation the run
    re-solves: none unless ``run_perturbations`` and ``run_haemodynamics``
    are both on, the only way the perturbation stage runs any."""
    if not (settings.get("run_perturbations") and settings.get("run_haemodynamics")):
        return []
    entries = []
    for entry in settings.get("perturbations") or ():
        if isinstance(entry, Mapping):
            overrides = entry.get("overrides") or {}
            entries.append((str(entry.get("type")), {**settings, **overrides}))
    return entries


def _runs_non_sweep_perturbation(settings: Mapping[str, Any]) -> bool:
    """A single re-solve writes the baseline's figures and statistics again
    for itself (``export_non_sweep_perturbation_artifacts``), whatever the
    baseline's own ``statistics`` and ``visualize_results`` say."""
    return any(not is_sweep_perturbation(kind) for kind, _ in _perturbation_entries(settings))


#: The two image readers, by file suffix: everything else the loaders take
#: is a TIFF.
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5"})

#: The large and small arteriole/venule mask paths, and each role's ilastik
#: switch (``pipeline.checks``' own pairing).
_VESSEL_MASK_FILES: dict[str, tuple[str, tuple[str, str]]] = {
    "large": (
        "use_ilastik_large_vessel_segmentation",
        ("large_arteriole_mask_path", "large_venule_mask_path"),
    ),
    "small": (
        "use_ilastik_small_vessel_segmentation",
        ("small_arteriole_mask_path", "small_venule_mask_path"),
    ),
}


def _reader(path: Any) -> str:
    return "h5py" if Path(str(path)).suffix.lower() in _HDF5_SUFFIXES else "tifffile"


def _image_readers(settings: Mapping[str, Any]) -> set[str]:
    """The readers (``"tifffile"``, ``"h5py"``) of every image file the run
    reads: the segmented image (ilastik's output when ilastik makes it), the
    raw image, the vessel masks in use, the tissue's own raw image, the 3D
    distance measurement's masks and the masks a perturbation reads. The raw
    images are read as TIFF whatever their name."""
    if _synthetic(settings):
        return set()
    readers: set[str] = set()
    ilastik_output = f"output{settings.get('ilastik_output_suffix') or '.tif'}"
    raw_image_set = _is_set(settings.get("raw_image_path"))
    if raw_image_set:
        readers.add("tifffile")
    if settings.get("use_ilastik_segmentation") and raw_image_set:
        readers.add(_reader(ilastik_output))
    elif _is_set(settings.get("input_path")):
        readers.add(_reader(settings.get("input_path")))
    for role, (ilastik_flag, mask_settings) in _VESSEL_MASK_FILES.items():
        if not vessel_masks_in_use(settings, role):
            continue
        if settings.get(ilastik_flag):
            readers.add(_reader(ilastik_output))
            continue
        readers.update(
            _reader(settings.get(name)) for name in mask_settings if _is_set(settings.get(name))
        )
    if settings.get("measure_tissue_volume") and _is_set(settings.get("tissue_raw_tiff_path")):
        readers.add("tifffile")
    if settings.get("measurement_3d_to_cell_mask"):
        readers.update(
            _reader(settings.get(name))
            for name in ("cell_mask_path", "measurement_3d_vessel_mask_path")
            if _is_set(settings.get(name))
        )
    if settings.get("measure_pericytes") and _is_set(settings.get("pericyte_measurement_mask_path")):
        readers.add(_reader(settings.get("pericyte_measurement_mask_path")))
    for kind, perturbed in _perturbation_entries(settings):
        if kind == "capillary_block_mask" and _is_set(perturbed.get("capillary_block_mask_path")):
            readers.add(_reader(perturbed.get("capillary_block_mask_path")))
        if (
            kind in (*_PERICYTE_CONSTRICTION_PERTURBATION_TYPES, "pericyte_measured_constriction")
            and perturbed.get("use_pericyte_mask_constriction")
            and _is_set(perturbed.get("pericyte_mask_path"))
        ):
            readers.add(_reader(perturbed.get("pericyte_mask_path")))
    return readers


def _used_hdf5(settings: Mapping[str, Any]) -> bool:
    return "h5py" in _image_readers(settings)


def _used_tifffile(settings: Mapping[str, Any]) -> bool:
    return "tifffile" in _image_readers(settings)


def _used_matplotlib(settings: Mapping[str, Any]) -> bool:
    """The run's own figures, or a perturbation's: a sweep draws its curves
    and a single re-solve its degree, overlay and resistance figures."""
    return bool(settings.get("visualize_results")) or bool(_perturbation_entries(settings))


def _used_plotly(settings: Mapping[str, Any]) -> bool:
    """The 3D views: the run's own and a single re-solve's in the ``3d``
    render mode, and the two vessel-mask debug views when their masks are in
    use."""
    if settings.get("final_render_mode") == "3d" and (
        settings.get("visualize_results") or _runs_non_sweep_perturbation(settings)
    ):
        return True
    if (
        settings.get("write_fast_mode_preassignment_large_vessel_debug_3d_html")
        and settings.get("automated_vessel_assignment_fast_mode")
        and vessel_masks_in_use(settings, "large")
    ):
        return True
    return bool(
        settings.get("write_small_vessel_boundary_labelling_3d_html")
        and vessel_masks_in_use(settings, "small")
    )


def _used_vtk_export(settings: Mapping[str, Any]) -> bool:
    """The vessel, pericyte and node files, and the tissue's surface."""
    return bool(
        (settings.get("run_haemodynamics") and settings.get("vtk_export"))
        or settings.get("measure_tissue_volume")
    )


def _pries_form(settings: Mapping[str, Any]) -> str | None:
    """The run's ``pries_viscosity_form``, or None when it uses no Pries law."""
    if not settings.get("run_haemodynamics") or settings.get("viscosity_law") != "pries":
        return None
    return settings.get("pries_viscosity_form") or "auto"


def _used_pries_in_vitro(settings: Mapping[str, Any]) -> bool:
    # The 2005 model is the in vitro law at the diameter its layer leaves free.
    return _pries_form(settings) in {"auto", "in_vitro_1992", "in_vivo_2005"}


def _used_pries_in_vivo_1994(settings: Mapping[str, Any]) -> bool:
    return _pries_form(settings) in {"auto", "in_vivo_1994"}


def _used_pries_secomb_2005(settings: Mapping[str, Any]) -> bool:
    return _pries_form(settings) == "in_vivo_2005"


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


def _used_red_cell_transport(settings: Mapping[str, Any]) -> bool:
    """Whether the run predicts tube haematocrit, red-cell velocity and
    flux (haemodynamics.red_cells): the flux heterogeneity statistic, or a
    capillary-block perturbation's comparison, on a solved network."""
    if not settings.get("run_haemodynamics"):
        return False
    if (
        settings.get("statistics")
        and settings.get("statistics_network_analysis")
        and settings.get("statistics_flux_heterogeneity")
    ):
        return True
    if not settings.get("run_perturbations"):
        return False
    entries = settings.get("perturbations") or ()
    return any(
        str(entry.get("type")) in ("capillary_block", "capillary_block_mask")
        for entry in entries
    )


def _used_open_end_estimate(settings: Mapping[str, Any]) -> bool:
    return bool(settings.get("run_haemodynamics")) and settings.get("open_end_model") == "estimated"


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
            (settings.get("do_skeletonize", True) and not _synthetic(settings))
            or settings.get("measure_tissue_volume")
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
        applies=lambda settings: bool(settings.get("do_graph_building", True)) and not _synthetic(settings),
    ),
    Citation(
        name="pandas",
        # Nothing the pipeline writes goes through pandas (the statistics
        # CSVs are written by the csv module); napari keeps each layer's
        # per-vessel features in a pandas table.
        reason="the per-vessel feature tables of the napari layers",
        reference=(
            "McKinney, W. (2010). Data structures for statistical "
            "computing in Python. Proceedings of the 9th Python in Science "
            "Conference, 56-61."
        ),
        applies=_used_napari,
    ),
    Citation(
        name="Matplotlib",
        reason="the static overlay, degree-distribution and perturbation plots",
        reference=(
            "Hunter, J.D. (2007). Matplotlib: a 2D graphics environment. "
            "Computing in Science & Engineering, 9(3), 90-95. "
            "https://doi.org/10.1109/MCSE.2007.55"
        ),
        applies=_used_matplotlib,
    ),
    Citation(
        name="Plotly",
        reason="the interactive 3D graph and vessel-mask views",
        reference=(
            "Plotly Technologies Inc. (2015). Collaborative data science. "
            "Plotly Technologies Inc., Montreal, QC. https://plot.ly"
        ),
        applies=_used_plotly,
    ),
    Citation(
        name="PyVista / VTK",
        reason="the exported .vtp vessel/pericyte/node and tissue-surface files",
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
        reason="loading the HDF5 images and masks",
        reference="Collette, A. (2013). Python and HDF5. O'Reilly Media.",
        applies=_used_hdf5,
    ),
    Citation(
        name="tifffile",
        reason="loading the TIFF images and masks",
        reference=(
            "Gohlke, C. tifffile: read and write TIFF files with Python. "
            "https://github.com/cgohlke/tifffile"
        ),
        applies=_used_tifffile,
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
        name="Pries in vitro viscosity law (Pries et al., 1992)",
        reason=(
            "apparent blood viscosity in a tube of the measured bore: plasma-column "
            "diameters, and the free lumen of the 2005 in vivo model"
        ),
        reference=(
            "Pries, A.R., Neuhaus, D., Gaehtgens, P. (1992). Blood "
            "viscosity in tube flow: dependence on diameter and "
            "hematocrit. American Journal of Physiology, 263(6 Pt 2), "
            "H1770-H1778."
        ),
        applies=_used_pries_in_vitro,
    ),
    Citation(
        name="Pries in vivo viscosity law (Pries et al., 1994)",
        reason="apparent blood viscosity in vivo, at wall-to-wall diameters",
        reference=(
            "Pries, A.R., Secomb, T.W., Gessner, T. et al. (1994). "
            "Resistance to blood flow in microvessels in vivo. "
            "Circulation Research, 75(5), 904-915."
        ),
        applies=_used_pries_in_vivo_1994,
    ),
    Citation(
        name="Pries-Secomb in vivo viscosity law (Pries and Secomb, 2005)",
        reason=(
            "apparent blood viscosity in vivo from the in vitro law and an "
            "endothelial surface layer whose thickness depends on diameter and haematocrit"
        ),
        reference=(
            "Pries, A.R., Secomb, T.W. (2005). Microvascular blood "
            "viscosity in vivo and the endothelial surface layer. "
            "American Journal of Physiology - Heart and Circulatory "
            "Physiology, 289(6), H2657-H2664."
        ),
        applies=_used_pries_secomb_2005,
    ),
    Citation(
        name="Legacy capillary power-law viscosity (viscosity_law='capillary_power_law')",
        reason="the placeholder viscosity model kept for comparison with earlier results",
        reference=(
            "This pipeline's own one-point calibration, 3.0 mPa.s at 5 um "
            "(see haemolynx.haemodynamics.viscosity); no published source is "
            "recorded for it, and it matches neither Pries form."
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
        name="Fahraeus effect: tube haematocrit from discharge haematocrit (Pries et al., 1990)",
        reason=(
            "each vessel's tube haematocrit, red-cell velocity, red-cell flux "
            "and spacing, predicted from the solved flow (the flux heterogeneity "
            "statistic and the capillary-block comparison)"
        ),
        reference=(
            "Pries, A.R., Secomb, T.W., Gaehtgens, P., Gross, J.F. (1990). "
            "Blood flow in microvascular networks. Experiments and simulation. "
            "Circulation Research, 67(4), 826-834."
        ),
        applies=_used_red_cell_transport,
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
        name="Estimated boundary conditions at open ends (open_end_model='estimated')",
        reason=(
            "pressures at the vessels leaving the image, chosen to keep the network's "
            "pressures and wall shear stresses near target values"
        ),
        reference=(
            "Fry, B.C., Lee, J., Smith, N.P., Secomb, T.W. (2012). Estimation of "
            "blood flow rates in large microvascular networks. Microcirculation, "
            "19(6), 530-538."
        ),
        applies=_used_open_end_estimate,
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


def _used_centreline_smoothing(method: str) -> Callable[[Mapping[str, Any]], bool]:
    def applies(settings: Mapping[str, Any]) -> bool:
        return bool(
            settings.get("do_graph_building", True)
            and settings.get("smooth_centrelines", True)
            and settings.get("centreline_smoothing_method", "taubin") == method
            and not _synthetic(settings)
        )

    return applies


#: How the vessels' centrelines are drawn from the segmentation.
CENTRELINE_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="Lee's 3D medial-axis thinning (skeletonize method='lee')",
        reason="the vessel centrelines thinned from the segmented mask",
        reference=(
            "Lee, T.C., Kashyap, R.L., Chu, C.N. (1994). Building skeleton "
            "models via 3-D medial surface/axis thinning algorithms. CVGIP: "
            "Graphical Models and Image Processing, 56(6), 462-478. "
            "https://doi.org/10.1006/cgip.1994.1042"
        ),
        applies=lambda settings: bool(settings.get("do_skeletonize", True))
        and not _synthetic(settings),
    ),
    Citation(
        name="Taubin smoothing (centreline_smoothing_method='taubin')",
        reason="the voxel staircase taken out of each centreline, which sets its length",
        reference=(
            "Taubin, G. (1995). Curve and surface smoothing without shrinkage. "
            "Proceedings of the Fifth International Conference on Computer "
            "Vision, 852-857. https://doi.org/10.1109/ICCV.1995.466848"
        ),
        applies=_used_centreline_smoothing("taubin"),
    ),
    Citation(
        name="Chaikin corner cutting (centreline_smoothing_method='chaikin')",
        reason="the voxel staircase taken out of each centreline, which sets its length",
        reference=(
            "Chaikin, G.M. (1974). An algorithm for high-speed curve generation. "
            "Computer Graphics and Image Processing, 3(4), 346-349. "
            "https://doi.org/10.1016/0146-664X(74)90028-8"
        ),
        applies=_used_centreline_smoothing("chaikin"),
    ),
)


def _statistics_measure_runs(
    measure: str, *, network_analysis: bool
) -> Callable[[Mapping[str, Any]], bool]:
    """Whether the statistics compute ``statistics_{measure}``: the baseline's
    report (``statistics`` on) or a single re-solve's, which always computes
    the run's own selection. A network analysis also needs
    ``statistics_network_analysis`` (``pipeline.stages.statistics_arguments``);
    *network_analysis* says which kind *measure* is, and a test pins it to
    ``statistics.NETWORK_ANALYSIS_MEASURES``."""

    def applies(settings: Mapping[str, Any]) -> bool:
        if not (settings.get("statistics") or _runs_non_sweep_perturbation(settings)):
            return False
        if not settings.get(f"statistics_{measure}"):
            return False
        return not network_analysis or bool(settings.get("statistics_network_analysis"))

    return applies


def _used_murray_law(settings: Mapping[str, Any]) -> bool:
    return _statistics_measure_runs("murray_law", network_analysis=False)(settings) or _synthetic(
        settings
    )


def _used_betweenness(settings: Mapping[str, Any]) -> bool:
    # The baseline's report always writes the length/resistance/flow-weighted
    # betweenness JSONs beside its CSV, whatever statistics_betweenness says.
    return bool(settings.get("statistics")) or _statistics_measure_runs(
        "betweenness", network_analysis=True
    )(settings)


def _used_greedy_modularity(settings: Mapping[str, Any]) -> bool:
    # The vessels layer's communities, and the weighted community counts the
    # baseline's report always writes, as well as the community measure.
    return (
        bool(settings.get("compute_vascular_communities"))
        or bool(settings.get("statistics"))
        or _statistics_measure_runs("community", network_analysis=True)(settings)
    )


#: Measures of the network graph, from the statistics report (and Murray's
#: law from a synthetic network's trees too).
NETWORK_STATISTICS_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="Murray's law",
        reason=(
            "the cube law relating a junction's parent and daughter diameters: "
            "the Murray's law statistic, and the diameters a synthetic network's "
            "trees are drawn with"
        ),
        reference=(
            "Murray, C.D. (1926). The physiological principle of minimum work: "
            "I. The vascular system and the cost of blood volume. Proceedings "
            "of the National Academy of Sciences, 12(3), 207-214. "
            "https://doi.org/10.1073/pnas.12.3.207"
        ),
        applies=_used_murray_law,
    ),
    Citation(
        name="Horton-Strahler ordering (statistics_strahler)",
        reason="the Strahler orders of the arterial and venous trees, and Horton's ratios",
        reference=(
            "Horton, R.E. (1945). Erosional development of streams and their "
            "drainage basins; hydrophysical approach to quantitative "
            "morphology. Geological Society of America Bulletin, 56(3), "
            "275-370. https://doi.org/10.1130/0016-7606(1945)56[275:EDOSAT]2.0.CO;2\n"
            "    Strahler, A.N. (1957). Quantitative analysis of watershed "
            "geomorphology. Transactions, American Geophysical Union, 38(6), "
            "913-920. https://doi.org/10.1029/TR038i006p00913"
        ),
        applies=_statistics_measure_runs("strahler", network_analysis=True),
    ),
    Citation(
        name="Greedy modularity communities (Clauset-Newman-Moore)",
        reason="the vascular communities and the community counts",
        reference=(
            "Clauset, A., Newman, M.E.J., Moore, C. (2004). Finding community "
            "structure in very large networks. Physical Review E, 70(6), "
            "066111. https://doi.org/10.1103/PhysRevE.70.066111"
        ),
        applies=_used_greedy_modularity,
    ),
    Citation(
        name="Kaplan-Meier estimator (statistics_capillary_bed_joins)",
        reason="the median branch order and distance at which capillary beds join",
        reference=(
            "Kaplan, E.L., Meier, P. (1958). Nonparametric estimation from "
            "incomplete observations. Journal of the American Statistical "
            "Association, 53(282), 457-481. "
            "https://doi.org/10.1080/01621459.1958.10501452"
        ),
        applies=_statistics_measure_runs("capillary_bed_joins", network_analysis=True),
    ),
    Citation(
        name="Algebraic connectivity (statistics_algebraic_connectivity)",
        reason="the Fiedler value of the network and the Fiedler split",
        reference=(
            "Fiedler, M. (1973). Algebraic connectivity of graphs. Czechoslovak "
            "Mathematical Journal, 23(2), 298-305. "
            "https://doi.org/10.21136/CMJ.1973.101168"
        ),
        applies=_statistics_measure_runs("algebraic_connectivity", network_analysis=True),
    ),
    Citation(
        name="Brandes' betweenness centrality",
        reason="the betweenness of each node, unweighted and weighted",
        reference=(
            "Brandes, U. (2001). A faster algorithm for betweenness centrality. "
            "Journal of Mathematical Sociology, 25(2), 163-177. "
            "https://doi.org/10.1080/0022250X.2001.9990249"
        ),
        applies=_used_betweenness,
    ),
)


def used_citations(
    settings: Mapping[str, Any], citations: tuple[Citation, ...]
) -> list[Citation]:
    """The entries in *citations* whose ``applies`` holds for *settings*."""
    return [citation for citation in citations if citation.applies(settings)]


#: What a synthetic network (``network_source=synthetic``) was drawn by.
NETWORK_GENERATION_CITATIONS: tuple[Citation, ...] = (
    Citation(
        name="Synthetic vascular network (network_source='synthetic')",
        reason=(
            "arterial and venous trees grown to the template run's branch-order "
            "table under Murray's law, and a capillary bed drawn as a "
            "degree-constrained random spatial network on jittered lattice "
            "points joined along Delaunay edges (or, with synthetic_bed_model "
            "voronoi, along the edges of a Voronoi tessellation of random "
            "points), matched to the template's capillary count, loops and "
            "dead ends; the organisational "
            "requirements a synthetic capillary bed is matched on follow"
        ),
        reference=(
            "Smith, A.F., Doyeux, V., Berg, M. et al. (2019). Brain capillary "
            "networks across species: a few simple organizational requirements "
            "are sufficient to reproduce both structure and function. Frontiers "
            "in Physiology, 10, 233. https://doi.org/10.3389/fphys.2019.00233"
        ),
        applies=_synthetic,
    ),
)


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
    centrelines = used_citations(settings, CENTRELINE_CITATIONS)
    if centrelines:
        lines.extend(_render_section("Centreline methods", centrelines, ""))
    image_analysis = used_citations(settings, IMAGE_ANALYSIS_CITATIONS)
    if image_analysis:
        lines.extend(_render_section("Image analysis methods", image_analysis, ""))
    network_statistics = used_citations(settings, NETWORK_STATISTICS_CITATIONS)
    if network_statistics:
        lines.extend(_render_section("Network statistics methods", network_statistics, ""))
    generation = used_citations(settings, NETWORK_GENERATION_CITATIONS)
    if generation:
        lines.extend(_render_section("Network generation methods", generation, ""))
    if vessel_classes_by_size(settings):
        title = "Vessel classes"
        lines.extend([
            title,
            "-" * len(title),
            "- Size convention (classify_vessels_by_size): each vessel classed by its "
            "measured width, mid at least "
            f"{settings.get('vessel_size_mid_min_diameter_um')} um and wide at least "
            f"{settings.get('vessel_size_wide_min_diameter_um')} um, thresholds stated for "
            "this run. No published rule: nothing to cite, state the thresholds instead.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_citations(settings: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render and write the citation text file for one run, returning its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_citations(settings), encoding="utf-8")
    return output_path
