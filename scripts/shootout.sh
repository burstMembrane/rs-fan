FILE="test_60s.wav"
CMD_ONE="./target/release/rs-fan $FILE"
CMD_TWO="./scripts/compare_ffmpeg.sh $FILE ffmpeg_output"
CMD_THREE="rs-fan $FILE --json"


hyperfine --warmup 3 --runs 10 "$CMD_ONE" "$CMD_TWO" "$CMD_THREE"

