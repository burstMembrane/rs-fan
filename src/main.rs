// Use jemalloc as the global allocator for better parallel performance (binary only)
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use anyhow::Result;
use clap::Parser;
use env_logger;
use log::info;
use std::path::PathBuf;

use rs_fan::{parse_format_spec, resample_fan, FormatSpec, Mp3Config, OutputFormat};

#[derive(Parser)]
#[command(name = "resamplefan")]
#[command(about = "A fast audio resampler using soxr")]
struct Args {
    /// Path(s) to the input audio file(s). Can specify multiple files or use glob patterns.
    input_files: Vec<PathBuf>,

    /// List of audio formats to process in RATE:CHANNELS format
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        "16000:1".to_string(),  // For Whisper + CREPE pipelines
        "44100:2".to_string(),  // For Demucs + Fingerprinter + HQ Vocal Separation
        "44100:1".to_string()   // For Essentia-based pipelines
    ])]
    formats: Vec<String>,

    /// Quality setting for resampling
    #[arg(long, default_value = "medium", value_parser = ["low", "medium", "high"])]
    quality: String,

    /// Number of SoXR threads (0 = auto)
    #[arg(long, default_value_t = 1)]
    soxr_threads: usize,

    /// Directory to save resampled files
    #[arg(long, default_value = "resampled")]
    output_dir: PathBuf,

    /// Output results as JSON
    #[arg(long, action = clap::ArgAction::SetTrue)]
    json: bool,

    /// Output format (wav or mp3)
    #[arg(long, default_value = "wav", value_parser = ["wav", "mp3"])]
    output_format: String,

    /// MP3 bitrate in kbps (64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
    #[arg(long, default_value_t = 192)]
    mp3_bitrate: u32,

    /// MP3 encoding quality (0 = best, 9 = worst)
    #[arg(long, default_value_t = 2)]
    mp3_quality: u32,

    /// Number of parallel encoding threads per MP3 file (0 = auto, 1 = single-threaded)
    #[arg(long, default_value_t = 4)]
    mp3_encoding_threads: usize,

    /// Number of parallel jobs for processing multiple files (0 = use all CPU cores)
    #[arg(long, short = 'j', default_value_t = 0)]
    jobs: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    env_logger::init();

    // Initialize rayon thread pool with configurable parallelism
    let num_threads = if args.jobs == 0 {
        num_cpus::get() // Use all available CPU cores
    } else {
        args.jobs
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok();

    if !args.json {
        info!("Using {} parallel job(s)", num_threads);
    }

    // Validate input files
    if args.input_files.is_empty() {
        return Err("No input files specified".into());
    }

    // Parse format specifications
    let specs: Result<Vec<FormatSpec>, _> =
        args.formats.iter().map(|s| parse_format_spec(s)).collect();
    let specs = specs?;

    // Parse output format
    let output_format = match args.output_format.as_str() {
        "wav" => OutputFormat::Wav,
        "mp3" => OutputFormat::Mp3,
        _ => return Err(format!("Invalid output format: {}", args.output_format).into()),
    };

    // Create MP3 config if needed
    let mp3_config = if output_format == OutputFormat::Mp3 {
        Some(Mp3Config {
            bitrate: args.mp3_bitrate,
            quality: args.mp3_quality,
            encoding_threads: args.mp3_encoding_threads,
        })
    } else {
        None
    };

    if !args.json {
        info!(
            "Processing {} file(s) with {} format(s) each",
            args.input_files.len(),
            specs.len()
        );
        if output_format == OutputFormat::Mp3 {
            info!(
                "Output format: MP3 ({}kbps, quality: {})",
                args.mp3_bitrate, args.mp3_quality
            );
        } else {
            info!("Output format: WAV");
        }
    }

    // Process all files in parallel using rayon
    use rayon::prelude::*;
    use rs_fan::BatchResult;

    let all_results: Vec<BatchResult> = args
        .input_files
        .par_iter()
        .map(|input_file| {
            if !args.json {
                info!("Processing: {}", input_file.display());
            }

            match resample_fan(
                input_file,
                specs.clone(),
                &args.output_dir,
                &args.quality,
                args.soxr_threads,
                output_format,
                mp3_config.clone(),
            ) {
                Ok(outputs) => BatchResult {
                    input_file: input_file.clone(),
                    outputs,
                    success: true,
                    error: None,
                },
                Err(e) => BatchResult {
                    input_file: input_file.clone(),
                    outputs: Vec::new(),
                    success: false,
                    error: Some(e.to_string()),
                },
            }
        })
        .collect();

    // Collect all output paths and handle errors
    let mut all_output_paths = Vec::new();
    let mut errors = Vec::new();

    for result in all_results {
        if result.success {
            all_output_paths.extend(result.outputs);
        } else {
            errors.push((
                result.input_file,
                result.error.unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }
    }

    // Display results
    if args.json {
        let json_output = serde_json::to_string_pretty(&all_output_paths)
            .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
        println!("{}", json_output);
    } else {
        for output_path in &all_output_paths {
            info!(
                " Created: {} | {}Hz {}ch",
                output_path.path.display(),
                output_path.format_spec.sample_rate,
                output_path.format_spec.channels
            );
        }
        info!(
            "\n Completed: {} output files from {} input file(s)",
            all_output_paths.len(),
            args.input_files.len()
        );

        if !errors.is_empty() {
            info!("\n Errors encountered:");
            for (file, error) in &errors {
                info!("  {}: {}", file.display(), error);
            }
            return Err(format!("{} file(s) failed to process", errors.len()).into());
        }
    }

    Ok(())
}
