# Saudi Female Speaker Reference

This directory contains the reference speaker audio for the Saudi XTTS TTS engine.

## Speaker Audio Requirements

| Attribute | Requirement |
|-----------|-------------|
| **Filename** | `reference.wav` |
| **Format** | WAV (PCM, 16-bit signed integer) |
| **Sample Rate** | 22050 Hz or 24000 Hz (XTTS native) |
| **Channels** | Mono (1 channel) |
| **Duration** | 6–15 seconds of clear speech |
| **Quality** | Studio/clean recording, minimal background noise |
| **Language** | Arabic (Saudi dialect preferred) |
| **Content** | Natural conversational text with varied intonation |
| **Speaker** | Female voice to be cloned |

## Best Practices for Speaker Reference

1. **Clear articulation**: Speak naturally but clearly
2. **Varied intonation**: Include questions, statements, emotional tones
3. **Minimal noise**: Record in a quiet environment
4. **No music/effects**: Pure voice only
5. **Consistent volume**: Avoid sudden volume changes

## Example Text (Arabic)

You can use this sample text to record your reference:

```
مرحباً بكم في عيادات قصي للتجميل والجلدية.
كيف يمكنني مساعدتك اليوم؟
نحن نقدم أفضل الخدمات التجميلية في المنطقة.
هل تريدين حجز موعد مع أحد أطبائنا المتخصصين؟
```

## Audio Processing

The reference audio will be processed by XTTS to extract:
- `gpt_cond_latent`: GPT conditioning latent for text generation
- `speaker_embedding`: Voice characteristics embedding

These are pre-computed at model load time for zero cold-start latency.

## Verification

After placing your `reference.wav` file, verify it works:

```bash
python backend/scripts/download_saudi_tts.py --check-speaker
```

## Technical Notes

- XTTS v2 uses the speaker reference to clone voice characteristics
- Longer references (10-15 seconds) generally produce better results
- The audio will be automatically resampled if needed (24kHz target)
