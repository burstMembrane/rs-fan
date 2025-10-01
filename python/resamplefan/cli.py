from typing import List

from resamplefan import resample_fan


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Resample an audio file using resamplefan."
    )
    parser.add_argument("input_file", help="Path to the input audio file.")
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
    args = parser.parse_args()
    # parse the formats
    formats = [
        {"sr": int(rate), "channels": int(channels)}
        for fmt in args.formats
        for rate, channels in [fmt.split(":")]
    ]
    result = resample_fan(
        audio_file=args.input_file,
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
