# rs-fan

A fast resampler than given an audio file, resamples to a number of different sample rate and channel specs


### Usage

This will output 2ch 44.1khz, 1ch 44.1khz 
```bash
rs-fan test_60s.wav --json --formats "44100:2,44100:1,16000:1,22050:2"
```