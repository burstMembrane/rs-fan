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
        type=List[str],
        default=["16000:1", "44100:2", "44100:1"],
        help="Output formats (e.g., wav, flac, mp3).",
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
    )
    print(result)


if __name__ == "__main__":
    main()
