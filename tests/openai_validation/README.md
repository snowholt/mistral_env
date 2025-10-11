# OpenAI API Validation Tests

**Purpose:** Validate OpenAI API integration for quality benchmarking of BeautyAI voice processing pipeline.

## Directory Structure

```
openai_validation/
├── .env                              # OpenAI API credentials
├── q7.webm                          # Test audio file (Arabic question)
├── test_openai_validation.py        # OpenAI API validation (Whisper + GPT-4o-mini)
├── test_websocket_integration.py    # WebSocket integration test with OpenAI comparison
├── cache/                           # Cached API results (auto-generated)
│   ├── q7_webm_transcription.json
│   └── response_*.json
├── validation_results.json          # OpenAI API test results
├── integration_test_results.json    # WebSocket integration test results
├── run_test.sh                      # Quick runner for validation test
└── README.md                        # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install openai python-dotenv
```

### 2. Configure API Key

The `.env` file already contains the API key. Verify it's present:

```bash
cat .env
```

## Usage

### Test 1: OpenAI API Validation (Standalone)

Tests OpenAI API independently:

```bash
cd /home/lumi/beautyai/tests/openai_validation
./run_test.sh
# or
python test_openai_validation.py
```

**What it does:**
1. Tests API connection
2. Transcribes q7.webm using **Whisper-1** (cheaper, cached)
3. Generates response using **GPT-4o-mini**
4. Saves results to cache and validation_results.json

**Caching:** Results are cached to avoid repeated API calls and costs. Delete `cache/` folder to force re-transcription.

---

### Test 2: Full WebSocket Integration Test

Tests the complete voice pipeline (BeautyAI WebSocket → STT → LLM → TTS) and validates quality by transcribing the TTS output back to OpenAI for comparison.

**Prerequisites:**
- BeautyAI backend API must be running: `python backend/run_server.py`
- Backend should be accessible at `ws://localhost:8000/api/v1/ws/simple-voice-chat`

**What it tests:**
1. WebSocket connection to BeautyAI backend
2. Audio upload (q7.webm)
3. Full pipeline: STT → LLM → TTS
4. TTS quality validation (transcribe back to OpenAI)
5. Similarity comparison between expected and actual responses

---

### Expected Output

```
======================================================================
🚀 OpenAI API Validation Pipeline
======================================================================

🔍 Testing API Connection...
✅ API Connection Successful
   Available models: XX found

🎤 Transcribing Audio: q7.webm
   File size: 46.00 KB
   Language: ar
✅ Transcription Successful (X.XXs)
   Detected Language: ar
   Transcribed Text: ...

🤖 Generating Response with GPT-4o-mini...
✅ Response Generated (X.XXs)
   Model: gpt-4o-mini-2024-07-18
   Tokens: XXX (prompt: XX, completion: XX)
   Response: ...

======================================================================
📊 VALIDATION SUMMARY
======================================================================
✅ API Connection: PASSED
✅ Transcription: PASSED
✅ Response Generation: PASSED

⏱️  Total Pipeline Time: X.XXs
   - Transcription: X.XXs
   - Response Generation: X.XXs

📝 Full Transcription:
   [Transcribed text here]

💬 AI Response:
   [AI response here]
======================================================================

💾 Results saved to: validation_results.json
```

## Results File

The script generates `validation_results.json` containing:

```json
{
  "transcription": {
    "success": true,
    "text": "...",
    "language": "ar",
    "duration": X.XX,
    "processing_time_sec": X.XX
  },
  "response": {
    "success": true,
    "response_text": "...",
    "model": "gpt-4o-mini-2024-07-18",
    "processing_time_sec": X.XX,
    "tokens": {
      "prompt": XX,
      "completion": XX,
      "total": XX
    }
  },
  "pipeline_total_time": X.XX,
  "success": true
}
```

## Use Cases

### 1. Quality Benchmarking
Compare OpenAI Whisper transcription quality vs. local Whisper models:
- Accuracy
- Language detection
- Processing speed

### 2. Response Quality Validation
Compare GPT-4o-mini responses vs. local Qwen model:
- Relevance
- Language quality
- Coherence

### 3. API Integration Testing
Verify OpenAI API is working before using it as reference standard.

## Notes

- **Audio Sample:** Tests use `voice_tests/input_test_questions/webm/q7.webm`
- **Language:** Default is Arabic (`ar`), can be changed in script
- **Model:** Uses `whisper-1` for STT and `gpt-4o-mini` for responses
- **Cost:** Each run consumes OpenAI API credits (minimal)

## Troubleshooting

### API Key Invalid
```
❌ API Connection Failed: Incorrect API key provided
```
**Solution:** Verify `.env` file contains correct key

### Missing Dependencies
```
❌ Missing dependencies. Install with:
   pip install openai python-dotenv
```
**Solution:** Run the pip install command shown

### Audio File Not Found
```
❌ Audio file not found: .../q7.webm
```
**Solution:** Verify `voice_tests/input_test_questions/webm/q7.webm` exists

## Security

⚠️ **Important:** 
- The `.env` file contains sensitive API credentials
- Do NOT commit `.env` to git
- Add to `.gitignore` if not already present
- Rotate API key after testing if exposed

---

**Last Updated:** 2025-10-06
