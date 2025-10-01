#!/bin/bash

# FFmpeg comparison script for resamplefan
# This script creates the same output files as our Rust CLI using ffmpeg
# Default formats from INPUT_SPECS.md:

set -e

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_audio_file> [output_dir]"
    echo "Example: $0 cometogether.wav ffmpeg_output"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-ffmpeg_resampled}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get input file basename without extension
BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')



# Function to run ffmpeg and measure time
run_ffmpeg() {
    local rate=$1
    local channels=$2
    local output_file=$3
    local description=$4
    local start_time=$(date +%s%N)

    if [ "$channels" == "1" ]; then
        # Mono output
        ffmpeg -i "$INPUT_FILE" -ar "$rate" -ac 1 -c:a pcm_f32le -y "$output_file" 2>/dev/null
    else
        # Stereo output
        ffmpeg -i "$INPUT_FILE" -ar "$rate" -ac 2 -c:a pcm_f32le -y "$output_file" 2>/dev/null
    fi

  
}


run_ffmpeg 16000 1 "$OUTPUT_DIR/${BASENAME}_16000Hz_1ch.wav" "16kHz mono"
run_ffmpeg 44100 2 "$OUTPUT_DIR/${BASENAME}_44100Hz_2ch.wav" "44.1kHz stereo"
run_ffmpeg 44100 1 "$OUTPUT_DIR/${BASENAME}_44100Hz_1ch.wav" "44.1kHz mono"

ls -lh "$OUTPUT_DIR"