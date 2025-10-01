use anyhow::Result;
use log::debug;
use rayon::prelude::*;
use serde::Serialize;
use sndfile::{
    Endian, MajorFormat, OpenOptions, ReadOptions, SndFileIO, SubtypeFormat, WriteOptions,
};
use soxr::format::{Mono, Stereo};
use soxr::{
    Soxr,
    params::{QualityRecipe, QualitySpec, RuntimeSpec},
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Python bindings module
pub mod python;

#[derive(Debug, Clone, Serialize)]
pub struct FormatSpec {
    pub sample_rate: u32,
    pub channels: usize,
}

#[derive(Debug, Serialize)]
pub struct OutputPath {
    pub path: PathBuf,
    pub format_spec: FormatSpec,
}

#[derive(Debug)]
pub struct WorkingBuffers {
    #[allow(dead_code)]
    mono_in: Vec<f32>, // input SR, mono
    resampled_mono: std::collections::HashMap<u32, Vec<f32>>, // sample_rate -> mono data
    resampled_stereo: std::collections::HashMap<u32, Vec<f32>>, // sample_rate -> stereo data
}

pub fn parse_format_spec(format_str: &str) -> Result<FormatSpec, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = format_str.split(':').collect();
    if parts.len() != 2 {
        return Err(
            format!("Invalid format specification: {format_str}. Expected RATE:CHANNELS").into(),
        );
    }

    let sample_rate = parts[0]
        .parse::<u32>()
        .map_err(|_| format!("Invalid sample rate: {}", parts[0]))?;
    let channels = parts[1]
        .parse::<usize>()
        .map_err(|_| format!("Invalid channel count: {}", parts[1]))?;

    Ok(FormatSpec {
        sample_rate,
        channels,
    })
}

pub fn get_quality_recipe(quality: &str) -> QualityRecipe {
    match quality {
        "low" => QualityRecipe::Low,
        "medium" => QualityRecipe::Medium,
        "high" | _ => QualityRecipe::Bits20,
    }
}

// Fast SIMD downmix: stereo -> mono
pub fn make_mono_simd(stereo: &[f32]) -> Vec<f32> {
    assert!(stereo.len() % 2 == 0);
    let n_frames = stereo.len() / 2;
    let mut out = vec![0.0f32; n_frames];

    // Process in chunks for better performance
    for (i, chunk) in stereo.chunks_exact(2).enumerate() {
        out[i] = 0.5 * (chunk[0] + chunk[1]);
    }

    out
}

// Fast stereo duplication: mono -> stereo
pub fn duplicate_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = vec![0.0f32; mono.len() * 2];
    for (i, &sample) in mono.iter().enumerate() {
        let j = i * 2;
        stereo[j] = sample;
        stereo[j + 1] = sample;
    }
    stereo
}

pub fn resample_audio(
    input: &[f32],
    in_sr: f64,
    out_sr: f64,
    input_channels: usize,
    output_channels: usize,
    quality: QualityRecipe,
    threads: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Skip resampling if sample rates match
    if (in_sr - out_sr).abs() < 0.1 {
        // Still need to handle channel conversion even if sample rate is the same
        return Ok(match (input_channels, output_channels) {
            (1, 1) => input.to_vec(), // mono to mono
            (1, 2) => duplicate_to_stereo(input), // mono to stereo
            (2, 1) => make_mono_simd(input), // stereo to mono
            (2, 2) => input.to_vec(), // stereo to stereo
            _ => return Err("Unsupported channel configuration".into()),
        });
    }

    let q = QualitySpec::new(quality);
    let rt = RuntimeSpec::new(threads as u32);

    // Pre-allocate with better estimation
    let ratio = out_sr / in_sr;
    let input_frames = input.len() / input_channels;
    let est_frames = ((input_frames as f64) * ratio).ceil() as usize + 1024;
    let est_samples = est_frames * output_channels;
    let mut out = vec![0f32; est_samples];

    let written = match (input_channels, output_channels) {
        (1, 1) => {
            // Mono to mono
            let mut soxr = Soxr::<Mono<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create mono resampler: {:?}", e))?;
            let processed = soxr.process(input, &mut out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;

            // Drain the mono resampler
            let mut written = processed.output_frames;
            loop {
                if written == out.len() {
                    out.resize(out.len() * 3 / 2, 0.0);
                }
                let n = soxr.drain(&mut out[written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 { break; }
                written += n;
            }
            written
        }
        (1, 2) => {
            // Mono to stereo: resample mono first, then duplicate
            let mut mono_out = vec![0f32; est_frames];
            let mut soxr = Soxr::<Mono<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create mono resampler: {:?}", e))?;
            let processed = soxr.process(input, &mut mono_out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;
            let mono_len = processed.output_frames;

            // Drain the mono resampler
            let mut mono_written = mono_len;
            loop {
                if mono_written == mono_out.len() {
                    mono_out.resize(mono_out.len() * 3 / 2, 0.0);
                }
                let n = soxr.drain(&mut mono_out[mono_written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 { break; }
                mono_written += n;
            }
            mono_out.truncate(mono_written);

            // Convert to stereo
            let stereo_result = duplicate_to_stereo(&mono_out);
            let stereo_len = stereo_result.len();
            if stereo_len <= out.len() {
                out[..stereo_len].copy_from_slice(&stereo_result);
            } else {
                out = stereo_result;
            }
            stereo_len
        }
        (2, 1) => {
            // Stereo to mono: downmix first, then resample mono
            let mono_input = make_mono_simd(input);
            let mut soxr = Soxr::<Mono<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create mono resampler: {:?}", e))?;
            let processed = soxr.process(&mono_input, &mut out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;
            let mut written = processed.output_frames;

            // Drain the mono resampler
            loop {
                if written == out.len() {
                    out.resize(out.len() * 3 / 2, 0.0);
                }
                let n = soxr.drain(&mut out[written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 { break; }
                written += n;
            }
            written
        }
        (2, 2) => {
            // Stereo to stereo: convert formats for soxr
            let input_frames = input.len() / 2;
            let mut input_stereo = vec![[0f32; 2]; input_frames];
            for (i, chunk) in input.chunks_exact(2).enumerate() {
                input_stereo[i] = [chunk[0], chunk[1]];
            }

            let output_frames = est_frames;
            let mut output_stereo = vec![[0f32; 2]; output_frames];

            let mut soxr = Soxr::<Stereo<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create stereo resampler: {:?}", e))?;
            let processed = soxr.process(&input_stereo, &mut output_stereo)
                .map_err(|e| format!("Failed to process stereo audio: {:?}", e))?;
            let mut written_frames = processed.output_frames;

            // Drain the stereo resampler
            loop {
                if written_frames == output_stereo.len() {
                    output_stereo.resize(output_stereo.len() * 3 / 2, [0f32; 2]);
                }
                let n = soxr.drain(&mut output_stereo[written_frames..])
                    .map_err(|e| format!("Failed to drain stereo resampler: {:?}", e))?;
                if n == 0 { break; }
                written_frames += n;
            }

            // Convert back to interleaved f32
            let written_samples = written_frames * 2;
            if written_samples > out.len() {
                out.resize(written_samples, 0.0);
            }
            for (i, frame) in output_stereo.iter().take(written_frames).enumerate() {
                out[i * 2] = frame[0];
                out[i * 2 + 1] = frame[1];
            }

            written_samples
        }
        _ => return Err("Unsupported channel configuration".into()),
    };

    // Note: Draining is already handled in each specific case above

    out.truncate(written);
    Ok(out)
}

pub fn write_audio_float(
    data: &[f32],
    output_path: &Path,
    sample_rate: u32,
    channels: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = OpenOptions::WriteOnly(WriteOptions::new(
        MajorFormat::WAV,
        SubtypeFormat::FLOAT, // Use FLOAT instead of PCM_32
        Endian::File,
        sample_rate as usize,
        channels,
    ))
    .from_path(output_path)
    .map_err(|e| {
        format!(
            "Failed to create output file {}: {e:?}",
            output_path.display()
        )
    })?;

    out.write_from_slice(data)
        .map_err(|e| format!("Failed to write audio data: {e:?}"))?;

    Ok(())
}

pub fn process_optimized_pipeline(
    data: &[f32],
    input_sr: f64,
    input_channels: usize,
    specs: &[FormatSpec],
    quality: QualityRecipe,
    threads: usize,
) -> Result<WorkingBuffers, Box<dyn std::error::Error>> {
    // Step 1: Create mono input if needed
    let mono_in = if input_channels == 2 {
        make_mono_simd(data)
    } else {
        data.to_vec()
    };

    // Step 2: Check if input format matches any requested format
    let input_rate = input_sr as u32;
    let mut resampled_mono = std::collections::HashMap::new();
    let mut resampled_stereo = std::collections::HashMap::new();

    // If input already matches a requested format, store it directly
    for spec in specs {
        if spec.sample_rate == input_rate {
            match (input_channels, spec.channels) {
                (1, 1) => {
                    // Input is mono, request is mono - direct copy
                    if !resampled_mono.contains_key(&input_rate) {
                        debug!("Input format {}Hz mono matches request - using direct copy", input_rate);
                        resampled_mono.insert(input_rate, data.to_vec());
                    }
                }
                (2, 2) => {
                    // Input is stereo, request is stereo - direct copy
                    if !resampled_stereo.contains_key(&input_rate) {
                        debug!("Input format {}Hz stereo matches request - using direct copy", input_rate);
                        resampled_stereo.insert(input_rate, data.to_vec());
                    }
                }
                (2, 1) => {
                    // Input is stereo, request is mono - downmix
                    if !resampled_mono.contains_key(&input_rate) {
                        debug!("Input format {}Hz stereo matches mono request - using downmix", input_rate);
                        resampled_mono.insert(input_rate, mono_in.clone());
                    }
                }
                (1, 2) => {
                    // Input is mono, request is stereo - duplicate
                    if !resampled_stereo.contains_key(&input_rate) {
                        debug!("Input format {}Hz mono matches stereo request - using duplication", input_rate);
                        resampled_stereo.insert(input_rate, duplicate_to_stereo(data));
                    }
                }
                _ => {} // Unsupported channel combinations
            }
        }
    }

    // Step 3: Analyze remaining specs that need resampling
    let mut rate_channel_needs: std::collections::HashMap<u32, (bool, bool)> = std::collections::HashMap::new();
    for spec in specs {
        // Skip if this exact format is already handled
        if spec.sample_rate == input_rate {
            match spec.channels {
                1 if resampled_mono.contains_key(&spec.sample_rate) => continue,
                2 if resampled_stereo.contains_key(&spec.sample_rate) => continue,
                _ => {}
            }
        }

        let entry = rate_channel_needs.entry(spec.sample_rate).or_insert((false, false));
        match spec.channels {
            1 => entry.0 = true,  // needs mono
            2 => entry.1 = true,  // needs stereo
            _ => return Err("Unsupported channel count".into()),
        }
    }

    // If no resampling is needed, return early
    if rate_channel_needs.is_empty() {
        return Ok(WorkingBuffers {
            mono_in,
            resampled_mono,
            resampled_stereo,
        });
    }

    // Step 3: Determine optimal resampling strategy for each rate
    let resample_tasks: Vec<(u32, usize, usize)> = rate_channel_needs
        .iter()
        .map(|(&rate, &(needs_mono, needs_stereo))| {
            let (input_ch, output_ch) = match (needs_mono, needs_stereo) {
                (true, true) => (input_channels, 2), // Resample to stereo, derive mono
                (true, false) => (input_channels, 1), // Resample to mono only
                (false, true) => (input_channels, 2), // Resample to stereo only
                (false, false) => unreachable!(), // Should never happen
            };
            (rate, input_ch, output_ch)
        })
        .collect();

    // Step 4: Perform resampling (parallel processing)
    let resample_results: Vec<(u32, usize, Vec<f32>)> = resample_tasks
        .par_iter()
        .map(|&(rate, input_ch, output_ch)| {
            let input_data = if input_ch == input_channels {
                data
            } else if input_ch == 1 && input_channels == 2 {
                &mono_in
            } else {
                return Err(format!("Invalid channel configuration for rate {}", rate));
            };

            let result = resample_audio(
                input_data,
                input_sr,
                rate as f64,
                input_ch,
                output_ch,
                quality,
                threads,
            )
            .map_err(|e| format!("Failed to resample to {}Hz: {}", rate, e))?;

            Ok((rate, output_ch, result))
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Step 5: Merge resampling results with already-handled formats
    for (rate, output_channels, data) in resample_results {
        let (needs_mono, _needs_stereo) = rate_channel_needs[&rate];

        match output_channels {
            1 => {
                // We resampled to mono
                resampled_mono.insert(rate, data);
            }
            2 => {
                // We resampled to stereo
                resampled_stereo.insert(rate, data.clone());

                // If we also need mono for this rate, derive it from stereo
                if needs_mono {
                    let mono_data = make_mono_simd(&data);
                    resampled_mono.insert(rate, mono_data);
                }
            }
            _ => return Err("Unexpected output channel count".into()),
        }
    }

    Ok(WorkingBuffers {
        mono_in,
        resampled_mono,
        resampled_stereo,
    })
}

pub fn write_format_outputs(
    working: &WorkingBuffers,
    specs: &[FormatSpec],
    output_dir: &Path,
    input_stem: &str,
) -> Vec<Result<OutputPath, String>> {
    specs
        .par_iter()
        .map(|spec| {
            let start_time = Instant::now();

            let output_filename = format!(
                "{}_{:}Hz_{}ch.wav",
                input_stem, spec.sample_rate, spec.channels
            );
            let output_path = output_dir.join(output_filename);

            // Select the appropriate buffer based on channel count
            let output_data = match spec.channels {
                1 => {
                    working.resampled_mono.get(&spec.sample_rate)
                        .ok_or_else(|| format!("Mono data for {}Hz not found", spec.sample_rate))?
                        .clone()
                }
                2 => {
                    working.resampled_stereo.get(&spec.sample_rate)
                        .ok_or_else(|| format!("Stereo data for {}Hz not found", spec.sample_rate))?
                        .clone()
                }
                _ => return Err("Unsupported channel count".into()),
            };

            // Write to file
            write_audio_float(&output_data, &output_path, spec.sample_rate, spec.channels)
                .map_err(|e| format!("Write error: {}", e))?;

            let _duration = start_time.elapsed();
            debug!("Created file: {} ({}Hz {}ch) in {:.2}ms",
                   output_path.display(), spec.sample_rate, spec.channels,
                   _duration.as_secs_f64() * 1000.0);

            Ok(OutputPath {
                path: output_path,
                format_spec: spec.clone(),
            })
        })
        .collect()
}

/// Core resample_fan function that can be called from both CLI and Python
pub fn resample_fan(
    input_file: &Path,
    formats: Vec<FormatSpec>,
    output_dir: &Path,
    quality: &str,
    soxr_threads: usize,
) -> Result<Vec<OutputPath>, Box<dyn std::error::Error>> {
    // Create output directory
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output directory: {e}"))?;

    // Load input
    let mut snd = OpenOptions::ReadOnly(ReadOptions::Auto)
        .from_path(input_file)
        .map_err(|e| format!("Failed to open input file: {e:?}"))?;
    let sr_in = snd.get_samplerate() as f64;
    let n_channels = snd.get_channels() as usize;

    // Read all samples as f32 interleaved
    let data: Vec<f32> = snd
        .read_all_to_vec()
        .map_err(|e| format!("Failed to read audio data: {e:?}"))?;

    // Get quality setting
    let quality_recipe = get_quality_recipe(quality);

    // Get input filename stem
    let input_stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");

    // Process optimized pipeline
    let working = process_optimized_pipeline(
        &data,
        sr_in,
        n_channels,
        &formats,
        quality_recipe,
        soxr_threads,
    )?;

    // Write outputs
    let results = write_format_outputs(&working, &formats, output_dir, input_stem);

    // Collect successful outputs
    let mut output_paths = Vec::new();
    for result in results {
        match result {
            Ok(output_path) => output_paths.push(output_path),
            Err(e) => return Err(e.into()),
        }
    }

    Ok(output_paths)
}
