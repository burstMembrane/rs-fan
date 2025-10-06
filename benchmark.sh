CMD_ONE=".venv/bin/rs-fan test_60s.wav --output-format wav"
CMD_TWO=".venv/bin/rs-fan test_60s.wav --output-format mp3"

hyperfine --warmup 3 --runs 10 "$CMD_ONE"
hyperfine --warmup 3 --runs 10 "$CMD_TWO"
