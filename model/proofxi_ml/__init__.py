"""Football Proof AI's reproducible football model pipeline.

The package keeps data collection and as-of feature generation usable with the
Python standard library. Heavy training dependencies are imported only by the
training and inference commands.
"""

from .domain import Fixture
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureEngine

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FeatureEngine",
    "Fixture",
]
