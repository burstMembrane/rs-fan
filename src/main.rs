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
            "Processing {} format(s) from: {}",
            specs.len(),
            args.input_file.display()
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

    // Call core resample_fan function
    let output_paths = resample_fan(
        &args.input_file,
        specs.clone(),
        &args.output_dir,
        &args.quality,
        args.soxr_threads,
        output_format,
        mp3_config,
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
