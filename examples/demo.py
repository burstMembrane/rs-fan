"""
Example of using the resamplefan Python package.
This package resamples an input audio file to multiple sample rate and channel configurations using the soxr library.
It uses bindings to a Rust implementation of soxr for performance.
"""

from resamplefan import resample_fan

# your input audio
AUDIO_PATH = "test_60s.wav"
# where to save resampled files to
OUTPUT_DIR = "resampled"
# formats to resample to, must be a list of dicts with "sr" and "channels" keys
FORMATS = [
    {"sr": 16000, "channels": 1},
    {"sr": 44100, "channels": 2},
    {"sr": 44100, "channels": 1},
]
# soxr quality, one of "low", "medium", "high"
QUALITY = "medium"
# number of soxr threads to use (0 means auto)
SOXR_THREADS = 1

result = resample_fan(
    audio_file=AUDIO_PATH,
    formats=FORMATS,
    output_dir=OUTPUT_DIR,
    quality=QUALITY,
    soxr_threads=SOXR_THREADS,
)

# json output like
"""
[
  {
    "path": "resampled/test_60s_16000Hz_1ch.wav",
    "format_spec": {
      "sample_rate": 16000,
      "channels": 1
    }
  },
...
]
"""
print(result)
