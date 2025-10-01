"""
resamplefan - Fast audio resampling using SoXR

This module provides a Python interface to the resamplefan audio resampling library,
which uses the high-quality SoXR (SoX Resampler) library.
"""

from .resamplefan import resample_fan

__all__ = ["resample_fan"]
__version__ = "0.1.0"
