use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use crate::{resample_fan as core_resample_fan, FormatSpec};

/// Python wrapper for resample_fan
///
/// Args:
///     audio_file (str): Path to the input audio file
///     formats (list[dict]): List of format specifications, each dict with 'sr' and 'channels' keys
///         Example: [{"sr": 44100, "channels": 2}, {"sr": 16000, "channels": 1}]
///     output_dir (str): Directory to save resampled files (default: "resampled")
///     quality (str): Quality setting - "low", "medium", or "high" (default: "medium")
///     soxr_threads (int): Number of SoXR threads, 0 for auto (default: 1)
///
/// Returns:
///     str: JSON string containing list of output file paths and their format specs
///
/// Example:
///     >>> import resamplefan
///     >>> result = resamplefan.resample_fan(
///     ...     audio_file="input.wav",
///     ...     formats=[{"sr": 44100, "channels": 2}, {"sr": 16000, "channels": 1}],
///     ...     output_dir="output",
///     ...     quality="high",
///     ...     soxr_threads=4
///     ... )
///     >>> print(result)
#[pyfunction]
#[pyo3(signature = (audio_file, formats, output_dir="resampled", quality="medium", soxr_threads=1))]
fn resample_fan(
    audio_file: &str,
    formats: &Bound<'_, PyList>,
    output_dir: &str,
    quality: &str,
    soxr_threads: usize,
) -> PyResult<String> {
    // Parse formats from Python list of dicts
    let mut format_specs = Vec::new();

    for item in formats.iter() {
        let dict = item.downcast::<PyDict>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Each format must be a dict: {}", e)
            ))?;

        let sr = dict.get_item("sr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key"))?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key"))?
            .extract::<u32>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("'sr' must be an integer: {}", e)
            ))?;

        let channels = dict.get_item("channels")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key"))?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key"))?
            .extract::<usize>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("'channels' must be an integer: {}", e)
            ))?;

        format_specs.push(FormatSpec {
            sample_rate: sr,
            channels,
        });
    }

    // Validate quality
    if !["low", "medium", "high"].contains(&quality) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid quality '{}'. Must be 'low', 'medium', or 'high'", quality)
        ));
    }

    // Convert paths
    let input_path = PathBuf::from(audio_file);
    let output_path = PathBuf::from(output_dir);

    // Call core function
    let output_paths = core_resample_fan(
        &input_path,
        format_specs,
        &output_path,
        quality,
        soxr_threads,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Resampling failed: {}", e)
    ))?;

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&output_paths)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("JSON serialization failed: {}", e)
        ))?;
    Ok(json)
}

/// A Python module for fast audio resampling using SoXR
#[pymodule]
fn resamplefan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample_fan, m)?)?;
    Ok(())
}
