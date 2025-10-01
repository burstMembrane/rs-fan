use anyhow::Result;
use log::debug;
use rayon::prelude::*;
use serde::Serialize;
use sndfile::{
    Endian, MajorFormat, OpenOptions, ReadOptions, SndFileIO, SubtypeFormat, WriteOptions,
};
use soxr::format::{Mono, Stereo};
use soxr::{
    params::{QualityRecipe, QualitySpec, RuntimeSpec},
    Soxr,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Python bindings module
pub mod python;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Wav,
    Mp3,
}

#[derive(Debug, Clone)]
pub struct Mp3Config {
    pub bitrate: u32,
    pub quality: u32, // 0-9, where 0 is best and 9 is worst
    pub encoding_threads: usize, // Number of parallel encoding threads per file (0 = auto)
}

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
    _threads: usize, // Deprecated: SOXR uses 1 thread, rayon handles parallelism
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Skip resampling if sample rates match
    if (in_sr - out_sr).abs() < 0.1 {
        // Still need to handle channel conversion even if sample rate is the same
        return Ok(match (input_channels, output_channels) {
            (1, 1) => input.to_vec(),             // mono to mono
            (1, 2) => duplicate_to_stereo(input), // mono to stereo
            (2, 1) => make_mono_simd(input),      // stereo to mono
            (2, 2) => input.to_vec(),             // stereo to stereo
            _ => return Err("Unsupported channel configuration".into()),
        });
    }

    let q = QualitySpec::new(quality);
    // Use 1 thread per SOXR instance; rayon handles parallelism across tasks
    let rt = RuntimeSpec::new(1);

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
            let processed = soxr
                .process(input, &mut out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;

            // Drain the mono resampler
            let mut written = processed.output_frames;
            loop {
                if written == out.len() {
                    out.resize(out.len() * 3 / 2, 0.0);
                }
                let n = soxr
                    .drain(&mut out[written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 {
                    break;
                }
                written += n;
            }
            written
        }
        (1, 2) => {
            // Mono to stereo: resample mono first, then duplicate
            let mut mono_out = vec![0f32; est_frames];
            let mut soxr = Soxr::<Mono<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create mono resampler: {:?}", e))?;
            let processed = soxr
                .process(input, &mut mono_out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;
            let mono_len = processed.output_frames;

            // Drain the mono resampler
            let mut mono_written = mono_len;
            loop {
                if mono_written == mono_out.len() {
                    mono_out.resize(mono_out.len() * 3 / 2, 0.0);
                }
                let n = soxr
                    .drain(&mut mono_out[mono_written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 {
                    break;
                }
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
            let processed = soxr
                .process(&mono_input, &mut out)
                .map_err(|e| format!("Failed to process mono audio: {:?}", e))?;
            let mut written = processed.output_frames;

            // Drain the mono resampler
            loop {
                if written == out.len() {
                    out.resize(out.len() * 3 / 2, 0.0);
                }
                let n = soxr
                    .drain(&mut out[written..])
                    .map_err(|e| format!("Failed to drain mono resampler: {:?}", e))?;
                if n == 0 {
                    break;
                }
                written += n;
            }
            written
        }
        (2, 2) => {
            // Stereo to stereo: optimized with pre-sized allocations
            let input_frames = input.len() / 2;

            // Pre-allocate with exact size needed
            let mut input_stereo = Vec::with_capacity(input_frames);
            unsafe {
                input_stereo.set_len(input_frames);
            }

            // Single-pass conversion using unsafe for speed
            let input_ptr = input.as_ptr();
            let output_ptr = input_stereo.as_mut_ptr() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(input_ptr, output_ptr, input.len());
            }

            let output_frames = est_frames;
            let mut output_stereo = vec![[0f32; 2]; output_frames];

            let mut soxr = Soxr::<Stereo<f32>>::new_with_params(in_sr, out_sr, q, rt)
                .map_err(|e| format!("Failed to create stereo resampler: {:?}", e))?;
            let processed = soxr
                .process(&input_stereo, &mut output_stereo)
                .map_err(|e| format!("Failed to process stereo audio: {:?}", e))?;
            let mut written_frames = processed.output_frames;

            // Drain the stereo resampler
            loop {
                if written_frames == output_stereo.len() {
                    output_stereo.resize(output_stereo.len() * 3 / 2, [0f32; 2]);
                }
                let n = soxr
                    .drain(&mut output_stereo[written_frames..])
                    .map_err(|e| format!("Failed to drain stereo resampler: {:?}", e))?;
                if n == 0 {
                    break;
                }
                written_frames += n;
            }

            // Convert back to interleaved - single pass with exact size
            let written_samples = written_frames * 2;
            if written_samples > out.len() {
                out.resize(written_samples, 0.0);
            }

            // Fast copy using unsafe
            let src_ptr = output_stereo.as_ptr() as *const f32;
            let dst_ptr = out.as_mut_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, written_samples);
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

// Helper to encode a single chunk of audio to MP3
fn encode_mp3_chunk(
    data: &[f32],
    sample_rate: u32,
    channels: usize,
    config: &Mp3Config,
) -> Result<Vec<u8>, String> {
    use mp3lame_encoder::{Builder, DualPcm, FlushNoGap, Mode};

    // Fast saturating f32->i16 conversion
    #[inline(always)]
    fn f32_to_i16(x: f32) -> i16 {
        let y = (x * 32767.0).clamp(-32768.0, 32767.0);
        y as i16
    }

    // Initialize LAME encoder for this chunk
    let mut b = Builder::new().ok_or("Failed to create LAME builder")?;
    b.set_num_channels(2)
        .map_err(|e| format!("Failed to set channels: {:?}", e))?;
    b.set_sample_rate(sample_rate)
        .map_err(|e| format!("Failed to set sample rate: {:?}", e))?;
    b.set_mode(Mode::JointStereo)
        .map_err(|e| format!("Failed to set mode: {:?}", e))?;

    let bitrate = match config.bitrate {
        320 => mp3lame_encoder::Bitrate::Kbps320,
        256 => mp3lame_encoder::Bitrate::Kbps256,
        224 => mp3lame_encoder::Bitrate::Kbps224,
        192 => mp3lame_encoder::Bitrate::Kbps192,
        160 => mp3lame_encoder::Bitrate::Kbps160,
        128 => mp3lame_encoder::Bitrate::Kbps128,
        112 => mp3lame_encoder::Bitrate::Kbps112,
        96 => mp3lame_encoder::Bitrate::Kbps96,
        80 => mp3lame_encoder::Bitrate::Kbps80,
        64 => mp3lame_encoder::Bitrate::Kbps64,
        _ => mp3lame_encoder::Bitrate::Kbps192,
    };
    b.set_brate(bitrate)
        .map_err(|e| format!("Failed to set bitrate: {:?}", e))?;

    let quality = match config.quality {
        0..=2 => mp3lame_encoder::Quality::Best,
        3..=4 => mp3lame_encoder::Quality::Good,
        _ => mp3lame_encoder::Quality::Ok,
    };
    b.set_quality(quality)
        .map_err(|e| format!("Failed to set quality: {:?}", e))?;

    let mut enc = b
        .build()
        .map_err(|e| format!("Failed to initialize LAME encoder: {:?}", e))?;

    // Encode this chunk
    const FRAMES: usize = 1152 * 16;
    let mut left = vec![0i16; FRAMES];
    let mut right = vec![0i16; FRAMES];
    let max_need = mp3lame_encoder::max_required_buffer_size(FRAMES);
    let mut mp3_buffer = Vec::<u8>::with_capacity(max_need);
    let mut result = Vec::new();

    match channels {
        1 => {
            let mut cursor = 0;
            while cursor < data.len() {
                let take = FRAMES.min(data.len() - cursor);
                for i in 0..take {
                    left[i] = f32_to_i16(data[cursor + i]);
                }
                let pcm = DualPcm {
                    left: &left[..take],
                    right: &left[..take],
                };
                mp3_buffer.clear();
                let n = enc
                    .encode(pcm, mp3_buffer.spare_capacity_mut())
                    .map_err(|e| format!("Failed to encode mono: {:?}", e))?;
                unsafe {
                    mp3_buffer.set_len(n);
                }
                result.extend_from_slice(&mp3_buffer);
                cursor += take;
            }
        }
        2 => {
            let mut cursor = 0;
            while cursor < data.len() {
                let frames_avail = (data.len() - cursor) / 2;
                if frames_avail == 0 {
                    break;
                }
                let take_frames = FRAMES.min(frames_avail);
                for i in 0..take_frames {
                    let idx = cursor + 2 * i;
                    left[i] = f32_to_i16(data[idx]);
                    right[i] = f32_to_i16(data[idx + 1]);
                }
                let pcm = DualPcm {
                    left: &left[..take_frames],
                    right: &right[..take_frames],
                };
                mp3_buffer.clear();
                let n = enc
                    .encode(pcm, mp3_buffer.spare_capacity_mut())
                    .map_err(|e| format!("Failed to encode stereo: {:?}", e))?;
                unsafe {
                    mp3_buffer.set_len(n);
                }
                result.extend_from_slice(&mp3_buffer);
                cursor += take_frames * 2;
            }
        }
        _ => return Err(format!("Unsupported channel count for MP3: {}", channels).into()),
    }

    // Flush encoder
    mp3_buffer.clear();
    let n = enc
        .flush::<FlushNoGap>(mp3_buffer.spare_capacity_mut())
        .map_err(|e| format!("Failed to flush: {:?}", e))?;
    unsafe {
        mp3_buffer.set_len(n);
    }
    result.extend_from_slice(&mp3_buffer);

    Ok(result)
}

fn write_mp3(
    data: &[f32],
    output_path: &Path,
    sample_rate: u32,
    channels: usize,
    config: &Mp3Config,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufWriter, Write};
    use rayon::prelude::*;

    // Determine if we should use parallel encoding
    let use_parallel = config.encoding_threads > 1 && data.len() > sample_rate as usize * channels * 10;

    if !use_parallel {
        // Small file or single-threaded: use original streaming approach
        let encoded = encode_mp3_chunk(data, sample_rate, channels, config)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        std::fs::write(output_path, &encoded)
            .map_err(|e| format!("Failed to write MP3 file {}: {}", output_path.display(), e))?;
        return Ok(());
    }

    // Large file: split into chunks and encode in parallel
    let chunk_duration_secs = 5; // 5-second chunks
    let chunk_size_samples = sample_rate as usize * channels * chunk_duration_secs;

    // Align to frame boundaries for clean splits
    let frame_size = 1152 * channels;
    let chunk_size = (chunk_size_samples / frame_size) * frame_size;

    let chunks: Vec<&[f32]> = data.chunks(chunk_size).collect();

    debug!(
        "Encoding {} with {} parallel threads ({} chunks of ~{}s each)",
        output_path.display(),
        config.encoding_threads,
        chunks.len(),
        chunk_duration_secs
    );

    // Encode chunks in parallel
    let encoded_chunks: Result<Vec<Vec<u8>>, String> = chunks
        .par_iter()
        .map(|chunk| encode_mp3_chunk(chunk, sample_rate, channels, config))
        .collect();

    let encoded_chunks = encoded_chunks.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Write all chunks to file
    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("Failed to create {}: {}", output_path.display(), e))?;
    let mut writer = BufWriter::new(file);

    for chunk in encoded_chunks {
        writer.write_all(&chunk)?;
    }
    writer.flush()?;

    Ok(())
}

pub fn write_audio(
    data: &[f32],
    output_path: &Path,
    sample_rate: u32,
    channels: usize,
    format: OutputFormat,
    mp3_config: Option<&Mp3Config>,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Wav => write_audio_float(data, output_path, sample_rate, channels),
        OutputFormat::Mp3 => {
            let config = mp3_config.ok_or("MP3 config required for MP3 output")?;
            write_mp3(data, output_path, sample_rate, channels, config)
        }
    }
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

    // If input already matches a requested format, check if we need to store it
    // We only clone/convert if we actually need resampling OR if we need multiple outputs
    let needs_input_rate_mono = specs
        .iter()
        .any(|s| s.sample_rate == input_rate && s.channels == 1);
    let needs_input_rate_stereo = specs
        .iter()
        .any(|s| s.sample_rate == input_rate && s.channels == 2);

    // Store input data if needed (minimize clones)
    if needs_input_rate_mono || needs_input_rate_stereo {
        match (
            input_channels,
            needs_input_rate_mono,
            needs_input_rate_stereo,
        ) {
            (1, true, false) => {
                // Input is mono, only need mono - share original data
                debug!(
                    "Input format {}Hz mono matches request - sharing data",
                    input_rate
                );
                resampled_mono.insert(input_rate, data.to_vec());
            }
            (1, false, true) | (1, true, true) => {
                // Input is mono, need stereo (or both)
                debug!("Input format {}Hz mono, duplicating to stereo", input_rate);
                let stereo_data = duplicate_to_stereo(data);
                resampled_stereo.insert(input_rate, stereo_data);
                if needs_input_rate_mono {
                    resampled_mono.insert(input_rate, data.to_vec());
                }
            }
            (2, false, true) => {
                // Input is stereo, only need stereo - share original data
                debug!(
                    "Input format {}Hz stereo matches request - sharing data",
                    input_rate
                );
                resampled_stereo.insert(input_rate, data.to_vec());
            }
            (2, true, false) => {
                // Input is stereo, only need mono
                debug!("Input format {}Hz stereo, downmixing to mono", input_rate);
                resampled_mono.insert(input_rate, mono_in.clone());
            }
            (2, true, true) => {
                // Input is stereo, need both
                debug!("Input format {}Hz stereo, storing both", input_rate);
                resampled_stereo.insert(input_rate, data.to_vec());
                resampled_mono.insert(input_rate, mono_in.clone());
            }
            _ => {}
        }
    }

    // Step 3: Analyze remaining specs that need resampling
    let mut rate_channel_needs: std::collections::HashMap<u32, (bool, bool)> =
        std::collections::HashMap::new();
    for spec in specs {
        // Skip if this exact format is already handled
        if spec.sample_rate == input_rate {
            match spec.channels {
                1 if resampled_mono.contains_key(&spec.sample_rate) => continue,
                2 if resampled_stereo.contains_key(&spec.sample_rate) => continue,
                _ => {}
            }
        }

        let entry = rate_channel_needs
            .entry(spec.sample_rate)
            .or_insert((false, false));
        match spec.channels {
            1 => entry.0 = true, // needs mono
            2 => entry.1 = true, // needs stereo
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
                (false, false) => unreachable!(),    // Should never happen
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
                // If we also need mono for this rate, derive it from stereo before moving
                if needs_mono {
                    let mono_data = make_mono_simd(&data);
                    resampled_mono.insert(rate, mono_data);
                }

                // We resampled to stereo - move (no clone!)
                resampled_stereo.insert(rate, data);
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
    output_format: OutputFormat,
    mp3_config: Option<&Mp3Config>,
) -> Vec<Result<OutputPath, String>> {
    specs
        .par_iter()
        .map(|spec| {
            let start_time = Instant::now();

            let extension = match output_format {
                OutputFormat::Wav => "wav",
                OutputFormat::Mp3 => "mp3",
            };

            let output_filename = format!(
                "{}_{:}Hz_{}ch.{}",
                input_stem, spec.sample_rate, spec.channels, extension
            );
            let output_path = output_dir.join(output_filename);

            // Select the appropriate buffer based on channel count (no clone, just borrow)
            let output_data: &[f32] = match spec.channels {
                1 => working
                    .resampled_mono
                    .get(&spec.sample_rate)
                    .ok_or_else(|| format!("Mono data for {}Hz not found", spec.sample_rate))?,
                2 => working
                    .resampled_stereo
                    .get(&spec.sample_rate)
                    .ok_or_else(|| format!("Stereo data for {}Hz not found", spec.sample_rate))?,
                _ => return Err("Unsupported channel count".into()),
            };

            // Write to file
            write_audio(
                output_data,
                &output_path,
                spec.sample_rate,
                spec.channels,
                output_format,
                mp3_config,
            )
            .map_err(|e| format!("Write error: {}", e))?;

            let _duration = start_time.elapsed();
            debug!(
                "Created file: {} ({}Hz {}ch) in {:.2}ms",
                output_path.display(),
                spec.sample_rate,
                spec.channels,
                _duration.as_secs_f64() * 1000.0
            );

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
    output_format: OutputFormat,
    mp3_config: Option<Mp3Config>,
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
    let results = write_format_outputs(
        &working,
        &formats,
        output_dir,
        input_stem,
        output_format,
        mp3_config.as_ref(),
    );

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
