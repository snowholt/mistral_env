# Single Sentence TTS-to-Whisper Accuracy Test

## Overview
This script tests the accuracy of TTS generation and transcription for a single Arabic sentence using the BeautyAI framework.

## Test Process
1. **Text-to-Speech**: Converts the test sentence to audio using OuteTTS with Arabic speaker profile
2. **Speech-to-Text**: Transcribes the generated audio back to text using Whisper
3. **Accuracy Analysis**: Compares original vs transcribed text with detailed metrics

## Test Sentence
```
"مرحباً بكم في عيادة الجمال المتطورة، حيث نقدم أحدث علاجات البشرة والوجه باستخدام تقنيات الذكاء الاصطناعي المتقدمة والليزر الطبي المعتمد عالمياً لضمان النتائج المثلى."
```

## Usage

### Basic Usage
```bash
cd /home/lumi/beautyai
python test_single_sentence_tts_whisper.py
```

### Prerequisites
1. **BeautyAI Framework**: Properly installed and configured
2. **OuteTTS**: `pip install outetts`
3. **Arabic Speaker Profile**: Must exist at `/home/lumi/beautyai/voice_tests/arabic_speaker_profiles/arabic_female_premium_19s.json`
4. **Whisper Model**: `whisper-large-v3-turbo-arabic` configured in model registry

### Output Structure
```
voice_tests/single_sentence_test/
├── arabic_test_sentence_YYYYMMDD_HHMMSS.wav     # Generated audio
└── accuracy_test_results_YYYYMMDD_HHMMSS.json   # Test results
```

## Metrics Provided
- **Character Accuracy**: Character-level similarity percentage
- **Word Accuracy**: Word-level similarity percentage  
- **Length Differences**: Character and word count differences
- **Exact Match**: Boolean indicating perfect transcription
- **Performance**: TTS generation time, STT transcription time
- **File Statistics**: Audio file size and metadata

## Expected Results
- **High Accuracy**: >90% for good TTS-STT pipeline
- **Performance**: <5s total processing time
- **Audio Quality**: Clear WAV file generation

## Troubleshooting
1. **Missing Speaker Profile**: Script will fall back to default OuteTTS voice
2. **Whisper Model Issues**: Check model registry configuration
3. **Audio Generation Fails**: Verify OuteTTS installation and dependencies
4. **Transcription Fails**: Check Whisper model loading and GPU availability

## Integration with BeautyAI
- Uses `AudioTranscriptionService` for Whisper integration
- Leverages model registry for Whisper model configuration
- Follows BeautyAI logging and error handling patterns
- Compatible with existing Arabic speaker profiles

## Customization
To test different sentences, modify the `TEST_SENTENCE` variable in the script:
```python
TEST_SENTENCE = "Your Arabic sentence here"
```
