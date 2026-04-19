import warnings
from ..haemodynamics import *

warnings.warn(
    "The 'ImageLynx.hemodynamics' module has been renamed to 'ImageLynx.haemodynamics' "
    "for spelling consistency. Please update your imports. "
    "This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)
