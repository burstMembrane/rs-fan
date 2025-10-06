from pathlib import Path

from resamplefan import resample_fan, resample_fan_batch, set_num_threads


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Resample audio file(s) using resamplefan."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Path(s) to the input audio file(s). Can specify multiple files.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="resampled", help="Path to the output directory."
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality of the resampling.",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1, help="Number of soxr threads to use."
    )
    parser.add_argument(
        "-j", "--json", action="store_true", help="Output results in JSON format."
    )
    parser.add_argument(
        "-f",
        "--formats",
        nargs="+",
        default=["16000:1", "44100:2", "44100:1"],
        help="Output formats specs in rate:channel format.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["wav", "mp3"],
        default="wav",
        help="Output file format.",
    )
    parser.add_argument(
        "--mp3-bitrate",
        type=int,
        default=320,
        help="MP3 bitrate in kbps (64, 80, 96, 112, 128, 160, 192, 224, 256, 320).",
    )
    parser.add_argument(
        "--mp3-quality",
        type=int,
        default=2,
        choices=range(10),
        help="MP3 encoding quality (0=best, 9=worst).",
    )
    parser.add_argument(
        "--mp3-encoding-threads",
        type=int,
        default=4,
        help="Number of parallel encoding threads per MP3 file (0=auto, 1=single-threaded).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel jobs for processing multiple files (0=use all CPU cores).",
    )
    args = parser.parse_args()
    if not all(paths := [Path(arg).exists() for arg in args.input_files]):
        parser.error("One or more input files do not exist.")
        print(
            ", ".join(
                str(p) for p, exists in zip(args.input_files, paths) if not exists
            )
        )

    # Set number of threads for parallel processing
    if args.jobs != 0 or len(args.input_files) > 1:
        try:
            set_num_threads(args.jobs)
        except RuntimeError:
            # Thread pool already set, ignore
            pass

    # parse the formats
    formats = [
        {"sr": int(rate), "channels": int(channels)}
        for fmt in args.formats
        for rate, channels in [fmt.split(":")]
    ]

    # Use batch processing if multiple files, otherwise single file processing
    if len(args.input_files) > 1:
        result = resample_fan_batch(
            audio_files=args.input_files,
            formats=formats,
            output_dir=args.output_dir,
            quality=args.quality,
            soxr_threads=args.threads,
            output_format=args.output_format,
            mp3_bitrate=args.mp3_bitrate,
            mp3_quality=args.mp3_quality,
            mp3_encoding_threads=args.mp3_encoding_threads,
        )
    else:
        result = resample_fan(
            audio_file=args.input_files[0],
            formats=formats,
            output_dir=args.output_dir,
            quality=args.quality,
            soxr_threads=args.threads,
            output_format=args.output_format,
            mp3_bitrate=args.mp3_bitrate,
            mp3_quality=args.mp3_quality,
            mp3_encoding_threads=args.mp3_encoding_threads,
        )
    print(result)


if __name__ == "__main__":
    main()
