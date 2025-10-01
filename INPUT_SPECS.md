# INPUT_SPECS.md

| Pipeline Name | Sample Rate | Channels | Notes |
|---------------|-------------|----------|-------|
| vocal_transcription | 16000 | 1 | Triton WhisperClient TARGET_SR = 16000 |
| source_sep | 44100 | 2 | Triton DemucsClient DEMUCS_SPEC requirement |
| vocal_melody_transcription | 16000 | 1 | Triton CrepeClient CREPE_SPEC requirement |
| key_analysis | 44100 | 1 | Essentia MonoLoader default (22050 for GPU mode) |
| chordino | 44100 | 1 | Preserves original SR, converts to mono |
| chord_transcription | 44100 | 1 | BTC model, preserves original SR |
| metronome | 44100 | 1 | Essentia Rhythm Extractor default |
| audio_fingerprinter | 44100 | 2 | Standard fingerprinting format |
| hq_vocal_separation | 44100 | 2 | SMSourceSeparator ML model standard |

## Preprocessing Strategy

From YouTube downloader's **44100Hz stereo WAV**, create these optimized formats:
- **16kHz mono**: For Whisper + CREPE pipelines
- **44.1kHz stereo**: For Demucs + Fingerprinter + HQ Vocal Separation
- **44.1kHz mono**: For Essentia-based pipelines

This eliminates redundant resampling in Triton clients and ensures optimal performance.