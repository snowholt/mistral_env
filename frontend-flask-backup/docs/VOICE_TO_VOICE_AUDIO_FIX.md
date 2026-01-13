# Voice-to-Voice Audio Issue Solution

## 🔍 Problem Analysis

The voice-to-voice feature is working correctly in terms of API communication and text processing, but the audio playback is not working. Here's what's happening:

### ✅ What's Working:
- Voice-to-voice API call succeeds
- Transcription works: `"هي ، كيف ستكون ؟"`
- Text response is generated: Arabic response about treatment results
- Audio size indicates generation: `516684 bytes`
- Payload format is now correct (no duplicates, proper booleans)

### ❌ What's NOT Working:
- **Audio playback in the web UI**
- The JavaScript `playVoiceResponse()` function is not being called
- No `audio_data` field in the API response

## 🔍 Root Cause

The BeautyAI voice-to-voice API is returning a **JSON response with metadata** but **NOT including the actual audio data** in the response. The web UI expects the audio to be in an `audio_data` field as base64-encoded content.

Looking at your successful response:
```json
{
    "success": true,
    "transcription": "هي ، كيف ستكون ؟",
    "response_text": "ستكون النتيجة متناسقة مع أهداف العلاج...",
    "audio_size_bytes": 516684,
    "audio_output_format": "wav",
    // MISSING: "audio_data": "base64_encoded_audio_here"
}
```

## 🛠️ Solutions

### Solution 1: Fix BeautyAI API to Include Audio Data (Recommended)

The BeautyAI voice-to-voice endpoint should be modified to include the generated audio in the JSON response:

```python
# In BeautyAI API voice-to-voice endpoint
import base64

# After generating audio_bytes from TTS
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

response = {
    "success": True,
    "transcription": transcription,
    "response_text": response_text,
    "audio_data": audio_base64,  # ADD THIS LINE
    "audio_output_format": "wav",
    "audio_size_bytes": len(audio_bytes),
    # ... other metadata
}
```

### Solution 2: Web UI Backend Fallback (Interim Fix)

If you can't modify the BeautyAI API immediately, you can add a fallback in the web UI backend to fetch the audio separately:

```python
# In src/web_ui/app.py - voice_to_voice endpoint
if result.get('success', False):
    response_data = {
        'success': True,
        'transcription': result.get('transcription', ''),
        'response_text': result.get('response_text', ''),
        # ... other fields
    }
    
    # If audio_data is missing but audio was generated, try to fetch it
    if 'audio_data' not in result and result.get('audio_size_bytes', 0) > 0:
        session_id = result.get('session_id', '')
        if session_id:
            # Try to fetch audio from a separate endpoint
            try:
                audio_url = f"{BEAUTYAI_API_URL}/audio/{session_id}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(audio_url) as audio_response:
                        if audio_response.status == 200:
                            audio_bytes = await audio_response.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                            response_data['audio_data'] = audio_base64
            except Exception as e:
                logger.warning(f"Could not fetch audio separately: {e}")
    
    # If we have audio data, include it
    elif 'audio_data' in result:
        response_data['audio_data'] = result['audio_data']
```

### Solution 3: JavaScript Fallback (Last Resort)

If neither above solution is possible, you could modify the JavaScript to try fetching audio separately:

```javascript
// In src/web_ui/static/js/main.js - processVoiceTurn method
if (result.success) {
    // ... existing code ...
    
    // Try to play audio if available
    if (result.audio_data) {
        await this.playVoiceResponse(result.audio_data, result.audio_output_format || 'wav');
    } else if (result.session_id && result.audio_size_bytes > 0) {
        // Try to fetch audio separately
        try {
            const audioResponse = await fetch(`/api/voice-audio/${result.session_id}`);
            if (audioResponse.ok) {
                const audioBlob = await audioResponse.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                // Play audio directly
                const audio = new Audio(audioUrl);
                await audio.play();
            }
        } catch (error) {
            console.warn('Could not fetch audio separately:', error);
        }
    }
}
```

## 🚀 Quick Test Commands

### 1. Test Current Voice-to-Voice Response
```bash
cd /home/lumi/benchmark_and_test
python debug_voice_to_voice_audio.py
```

### 2. Test Direct API Call
```bash
curl -X POST http://localhost:8000/inference/voice-to-voice \
  -F "audio_file=@test_audio.wav" \
  -F "input_language=auto" \
  -F "output_language=auto" \
  -F "preset=qwen_optimized" \
  -F "chat_model_name=qwen3-unsloth-q4ks"
```

### 3. Test Web UI Endpoint
```bash
curl -X POST http://localhost:5001/api/voice-to-voice \
  -F "audio_file=@test_audio.wav" \
  -F "input_language=auto" \
  -F "output_language=auto" \
  -F "preset=qwen_optimized"
```

## ✅ Payload Issues Fixed

Your payload format has been corrected:

### ❌ Before (Problematic):
```
preset=high_quality
preset=qwen_optimized                # Duplicate
disable_content_filter=on
disable_content_filter=true          # Duplicate  
content_filter_strictness=balanced
content_filter_strictness=balanced   # Duplicate
thinking_mode=false                  # String instead of boolean
input_language=ar                    # Hardcoded instead of auto
output_language=ar                   # Hardcoded instead of auto
```

### ✅ After (Fixed):
```
input_language=auto                  # Auto-detection
output_language=auto                 # Auto-detection
preset=qwen_optimized               # Single preset
disable_content_filter=true         # Boolean
content_filter_strictness=balanced  # Single entry
thinking_mode=true                  # Boolean
```

## 🎯 Immediate Action

1. **Check BeautyAI API logs** to see if TTS is generating audio correctly
2. **Run the debug script** to see exactly what the API returns
3. **Contact BeautyAI API team** to confirm if `audio_data` should be in the JSON response
4. **Implement Solution 1** by modifying the BeautyAI API to include audio_data

The payload formatting issues are now resolved, so the remaining issue is purely about the audio data not being included in the API response.
