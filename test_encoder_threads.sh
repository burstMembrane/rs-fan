

CMD=".venv/bin/rs-fan test_60s.wav --output-format mp3 --mp3-encoding-threads {threads}"


hyperfine --warmup 1 --runs 10 "$CMD" --parameter-list threads 1,2,4,8,16,32