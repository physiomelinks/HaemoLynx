"""A pixel classifier trained in the panel: HaemoLynx's own segmentation.

ilastik's way of working, inside HaemoLynx: paint a few strokes of each label
on the raw image, a random forest learns them from a bank of filters at
several scales (:mod:`.features`), and the saved classifier (:mod:`.model`, a
``.haemoclf`` file) segments any raw image of the same kind
(:mod:`.apply`), the main vessel mask on the Input tab or the arteriole and
venule masks on the Boundaries tab. Its outputs are 0/1 masks named and kept
as ilastik's are, so everything after segmentation reads them alike.

Training needs scikit-learn; reading and applying a classifier needs only
NumPy and SciPy.
"""
from .apply import (
    RawChannels,
    classifier_output_current,
    classifier_output_is_current,
    classifier_provenance,
    classify_volume,
    read_raw_channels,
    segment_with_classifier,
)
from .features import (
    DEFAULT_SCALES_UM,
    SCALE_CHOICES_UM,
    FeatureSpec,
    box_features,
    boxes,
    feature_halo,
    features_at,
    intensity_range,
)
from .forest import Forest, scikit_learn_available, train_forest
from .model import (
    FILE_SUFFIX,
    PURPOSES,
    VESSEL_LABEL,
    PixelClassifier,
    TrainingImage,
    TrainingVoxels,
    load_classifier,
    new_classifier,
    read_classifier_record,
    save_classifier,
)

__all__ = [
    "DEFAULT_SCALES_UM",
    "FILE_SUFFIX",
    "PURPOSES",
    "SCALE_CHOICES_UM",
    "VESSEL_LABEL",
    "FeatureSpec",
    "Forest",
    "PixelClassifier",
    "RawChannels",
    "TrainingImage",
    "TrainingVoxels",
    "box_features",
    "boxes",
    "classifier_output_current",
    "classifier_output_is_current",
    "classifier_provenance",
    "classify_volume",
    "feature_halo",
    "features_at",
    "intensity_range",
    "load_classifier",
    "new_classifier",
    "read_classifier_record",
    "read_raw_channels",
    "save_classifier",
    "scikit_learn_available",
    "segment_with_classifier",
    "train_forest",
]
