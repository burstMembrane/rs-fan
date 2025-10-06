#!/bin/bash

# Batch mode: process all files at once with internal parallelization
CMD_BATCH="./target/release/resamplefan stems/*.wav --formats 44100:2 --output-format mp3 --mp3-bitrate 320 --output-dir resampled_batch"

# GNU parallel mode: run separate instances for each file
CMD_PARALLEL="ls stems/*.wav | parallel -j 64 './target/release/resamplefan {} --formats 44100:2 --output-format mp3 --mp3-bitrate 320 --output-dir resampled_parallel'"

# FFmpeg with GNU parallel: convert each file with ffmpeg
CMD_FFMPEG="ls stems/*.wav | parallel -j 64 'ffmpeg -y -i {} -ar 44100 -ac 2 -b:a 320k resampled_ffmpeg/{/.}.mp3 -loglevel error'"

CMD_PY_BATCH="rs-fan stems/*.wav --formats 44100:2 --output-format mp3 --mp3-bitrate 320 --output-dir resampled_py_batch"
# Prepare output directories
mkdir -p resampled_batch resampled_parallel resampled_ffmpeg

echo "===================================================================="
echo "  Performance Benchmark: resamplefan vs FFmpeg"
echo "  Test: Convert 7 WAV files to 44.1kHz stereo MP3 @ 320kbps"
echo "  Hardware: AMD Threadripper 3970X (64 threads)"
echo "===================================================================="
echo ""

echo "1. Batch mode (internal parallelization - RECOMMENDED)..."
hyperfine --warmup 2 --runs 10 "$CMD_BATCH"

echo ""
echo "2. GNU parallel mode (separate instances)..."
hyperfine --warmup 2 --runs 10 "$CMD_PARALLEL"

echo ""
echo "3. FFmpeg with GNU parallel (baseline)..."
hyperfine --warmup 2 --runs 10 "$CMD_FFMPEG"

echo ""
echo "4. Python batch mode (internal parallelization)..."
hyperfine --warmup 2 --runs 10 "$CMD_PY_BATCH"
