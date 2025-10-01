use anyhow::Result;
use clap::Parser;
use log::{info, warn, error};
use std::path::PathBuf;
use env_logger;

use rs_fan::{parse_format_spec, FormatSpec, resample_fan};


#[derive(Parser)]
#[command(name = "resamplefan")]
#[command(about = "A fast audio resampler using soxr")]
struct Args {
    /// Path to the input audio file
    input_file: PathBuf,

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    env_logger::init();

    // Initialize rayon thread pool to avoid oversubscription
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(4, num_cpus::get()))
        .build_global()
        .ok();

    // Parse format specifications
    let specs: Result<Vec<FormatSpec>, _> =
        args.formats.iter().map(|s| parse_format_spec(s)).collect();
    let specs = specs?;

    if !args.json {
        info!(
            "Processing {} format(s) from: {}",
            specs.len(),
            args.input_file.display()
        );
    }

    // Call core resample_fan function
    let output_paths = resample_fan(
        &args.input_file,
        specs.clone(),
        &args.output_dir,
        &args.quality,
        args.soxr_threads,
    )?;

    // Display results
    if args.json {
        let json_output = serde_json::to_string_pretty(&output_paths)
            .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
        println!("{}", json_output);
    } else {
        for output_path in &output_paths {
            info!(
                " Created: {} | {}Hz {}ch",
                output_path.path.display(),
                output_path.format_spec.sample_rate,
                output_path.format_spec.channels
            );
        }
        info!(
            "\n Completed: {}/{} files processed successfully",
            output_paths.len(),
            specs.len()
        );
    }

    Ok(())
}
