# Chatterbox Speaker Reference Files

This directory contains reference audio files for Chatterbox TTS voice cloning.

## Requirements

- **Duration**: ~10-15 seconds of clear speech
- **Format**: WAV, 16-bit PCM, mono
- **Sample Rate**: 24kHz recommended (will be resampled if different)
- **Quality**: Clean audio, minimal background noise, consistent volume

## Files

- `reference.wav` - Default speaker reference for voice cloning

## Adding New Speakers

1. Record or prepare a 10-15 second audio clip of the target voice
2. Convert to WAV format:
   ```bash
   ffmpeg -i input.mp3 -ar 24000 -ac 1 -acodec pcm_s16le speaker_name.wav
   ```
3. Place in this directory
4. Update config to reference the new file

## Language Tips

For best results with Chatterbox Multilingual:

- **Arabic**: Use an Arabic speaker reference for Arabic synthesis
- **English**: Use an English speaker reference for English synthesis
- Cross-language cloning works but may inherit accent from reference

## Generation Parameters

- `exaggeration`: 0.0-1.0 (higher = more expressive, default 0.5)
- `cfg_weight`: 0.0-1.0 (lower = slower pacing, default 0.5)

For expressive/dramatic speech:
- exaggeration: 0.7
- cfg_weight: 0.3

## Model Info

- Model: ResembleAI/chatterbox (Multilingual variant)
- Parameters: 500M
- Languages: 23 (ar, en, fr, de, es, zh, ja, ko, etc.)
- HuggingFace: https://huggingface.co/ResembleAI/chatterbox
