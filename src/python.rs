use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

use crate::{
    resample_fan as core_resample_fan, resample_fan_batch as core_resample_fan_batch, FormatSpec,
    Mp3Config, OutputFormat,
};

/// Resample and optionally encode an audio file to multiple formats
///
/// Args:
///     audio_file (str): Path to the input audio file
///     formats (list[dict]): List of format specifications, each dict with 'sr' and 'channels' keys
///         Example: [{"sr": 44100, "channels": 2}, {"sr": 16000, "channels": 1}]
///     output_dir (str): Directory to save resampled files (default: "resampled")
///     quality (str): Quality setting - "low", "medium", or "high" (default: "medium")
///     soxr_threads (int): Number of SoXR threads, 0 for auto (default: 1)
///     output_format (str): Output format - "wav" or "mp3" (default: "wav")
///     mp3_bitrate (int): MP3 bitrate in kbps (default: 192)
///     mp3_quality (int): MP3 quality, 0=best to 9=worst (default: 2)
///     mp3_encoding_threads (int): Number of parallel encoding threads per MP3 file (default: 4)
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
///     ...     soxr_threads=4,
///     ...     output_format="mp3",
///     ...     mp3_bitrate=192
///     ... )
///     >>> print(result)
#[pyfunction]
#[pyo3(signature = (audio_file, formats, output_dir="resampled", quality="medium", soxr_threads=1, output_format="wav", mp3_bitrate=192, mp3_quality=2, mp3_encoding_threads=4))]
fn resample_fan(
    audio_file: &str,
    formats: &Bound<'_, PyList>,
    output_dir: &str,
    quality: &str,
    soxr_threads: usize,
    output_format: &str,
    mp3_bitrate: u32,
    mp3_quality: u32,
    mp3_encoding_threads: usize,
) -> PyResult<String> {
    // Parse formats from Python list of dicts
    let mut format_specs = Vec::new();

    for item in formats.iter() {
        let dict = item.downcast::<PyDict>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Each format must be a dict: {}",
                e
            ))
        })?;

        let sr = dict
            .get_item("sr")
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key")
            })?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key")
            })?
            .extract::<u32>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "'sr' must be an integer: {}",
                    e
                ))
            })?;

        let channels = dict
            .get_item("channels")
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key")
            })?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key")
            })?
            .extract::<usize>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "'channels' must be an integer: {}",
                    e
                ))
            })?;

        format_specs.push(FormatSpec {
            sample_rate: sr,
            channels,
        });
    }

    // Validate quality
    if !["low", "medium", "high"].contains(&quality) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid quality '{}'. Must be 'low', 'medium', or 'high'",
            quality
        )));
    }

    // Parse output format
    let format = match output_format {
        "wav" => OutputFormat::Wav,
        "mp3" => OutputFormat::Mp3,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid output_format '{}'. Must be 'wav' or 'mp3'",
                output_format
            )));
        }
    };

    // Create MP3 config if needed
    let mp3_config = if format == OutputFormat::Mp3 {
        Some(Mp3Config {
            bitrate: mp3_bitrate,
            quality: mp3_quality,
            encoding_threads: mp3_encoding_threads,
        })
    } else {
        None
    };

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
        format,
        mp3_config,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Resampling failed: {}", e))
    })?;

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&output_paths).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "JSON serialization failed: {}",
            e
        ))
    })?;
    Ok(json)
}

/// Batch resample and optionally encode multiple audio files to multiple formats in parallel
///
/// Args:
///     audio_files (list[str]): List of paths to input audio files
///     formats (list[dict]): List of format specifications, each dict with 'sr' and 'channels' keys
///         Example: [{"sr": 44100, "channels": 2}, {"sr": 16000, "channels": 1}]
///     output_dir (str): Directory to save resampled files (default: "resampled")
///     quality (str): Quality setting - "low", "medium", or "high" (default: "medium")
///     soxr_threads (int): Number of SoXR threads, 0 for auto (default: 1)
///     output_format (str): Output format - "wav" or "mp3" (default: "wav")
///     mp3_bitrate (int): MP3 bitrate in kbps (default: 192)
///     mp3_quality (int): MP3 quality, 0=best to 9=worst (default: 2)
///     mp3_encoding_threads (int): Number of parallel encoding threads per MP3 file (default: 4)
///
/// Returns:
///     str: JSON string containing list of batch results with output file paths and their format specs
///
/// Example:
///     >>> import resamplefan
///     >>> result = resamplefan.resample_fan_batch(
///     ...     audio_files=["input1.wav", "input2.wav", "input3.wav"],
///     ...     formats=[{"sr": 44100, "channels": 2}, {"sr": 16000, "channels": 1}],
///     ...     output_dir="output",
///     ...     quality="high",
///     ...     output_format="mp3",
///     ...     mp3_bitrate=192
///     ... )
///     >>> print(result)
#[pyfunction]
#[pyo3(signature = (audio_files, formats, output_dir="resampled", quality="medium", soxr_threads=1, output_format="wav", mp3_bitrate=192, mp3_quality=2, mp3_encoding_threads=4))]
fn resample_fan_batch(
    audio_files: &Bound<'_, PyList>,
    formats: &Bound<'_, PyList>,
    output_dir: &str,
    quality: &str,
    soxr_threads: usize,
    output_format: &str,
    mp3_bitrate: u32,
    mp3_quality: u32,
    mp3_encoding_threads: usize,
) -> PyResult<String> {
    // Parse audio files from Python list
    let mut input_files = Vec::new();
    for item in audio_files.iter() {
        let path_str = item.extract::<String>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Each audio file must be a string: {}",
                e
            ))
        })?;
        input_files.push(PathBuf::from(path_str));
    }

    // Parse formats from Python list of dicts
    let mut format_specs = Vec::new();

    for item in formats.iter() {
        let dict = item.downcast::<PyDict>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Each format must be a dict: {}",
                e
            ))
        })?;

        let sr = dict
            .get_item("sr")
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key")
            })?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'sr' key")
            })?
            .extract::<u32>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "'sr' must be an integer: {}",
                    e
                ))
            })?;

        let channels = dict
            .get_item("channels")
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key")
            })?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Format dict missing 'channels' key")
            })?
            .extract::<usize>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "'channels' must be an integer: {}",
                    e
                ))
            })?;

        format_specs.push(FormatSpec {
            sample_rate: sr,
            channels,
        });
    }

    // Validate quality
    if !["low", "medium", "high"].contains(&quality) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid quality '{}'. Must be 'low', 'medium', or 'high'",
            quality
        )));
    }

    // Parse output format
    let format = match output_format {
        "wav" => OutputFormat::Wav,
        "mp3" => OutputFormat::Mp3,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid output_format '{}'. Must be 'wav' or 'mp3'",
                output_format
            )));
        }
    };

    // Create MP3 config if needed
    let mp3_config = if format == OutputFormat::Mp3 {
        Some(Mp3Config {
            bitrate: mp3_bitrate,
            quality: mp3_quality,
            encoding_threads: mp3_encoding_threads,
        })
    } else {
        None
    };

    // Convert paths
    let output_path = PathBuf::from(output_dir);

    // Call core batch function
    let batch_results = core_resample_fan_batch(
        &input_files,
        format_specs,
        &output_path,
        quality,
        soxr_threads,
        format,
        mp3_config,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Batch resampling failed: {}",
            e
        ))
    })?;

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&batch_results).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "JSON serialization failed: {}",
            e
        ))
    })?;
    Ok(json)
}

/// Set the number of threads for parallel processing
///
/// Args:
///     num_threads (int): Number of threads to use (0 = use all CPU cores)
///
/// Note: This must be called before any processing functions. Once set, it cannot be changed.
#[pyfunction]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    let threads = if num_threads == 0 {
        num_cpus::get()
    } else {
        num_threads
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to set thread pool: {}. Note: Thread pool can only be set once.",
                e
            ))
        })?;

    Ok(())
}

/// A Python module for fast audio resampling using SoXR
#[pymodule]
fn resamplefan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample_fan, m)?)?;
    m.add_function(wrap_pyfunction!(resample_fan_batch, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
