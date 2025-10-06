"""
resamplefan - Fast audio resampling using SoXR

This module provides a Python interface to the resamplefan audio resampling library,
which uses the high-quality SoXR (SoX Resampler) library and lame for MP3 support.
"""

from .resamplefan import resample_fan, resample_fan_batch, set_num_threads

__all__ = ["resample_fan", "resample_fan_batch", "set_num_threads"]
__version__ = "0.2.0"
