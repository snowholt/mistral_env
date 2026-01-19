Looking at your codebase, let me explain the difference between the WebSocket and Voice-to-Voice endpoint:

## They are DIFFERENT endpoints with different purposes:

### 1. **Voice-to-Voice REST API Endpoint** (`/inference/voice-to-voice`)
- **Purpose**: One-time voice processing
- **How it works**: Upload audio file → get audio response back
- **Use case**: Single conversation turns, file-based processing
- **Input**: HTTP POST with audio file upload
- **Output**: JSON response with base64-encoded audio

### 2. **WebSocket Voice Endpoint** (`/ws/voice-conversation`) 
- **Purpose**: Real-time continuous conversation
- **How it works**: Persistent connection, send audio chunks, get real-time responses
- **Use case**: Live conversation, streaming audio, chat-like experience
- **Input**: Binary WebSocket messages (audio chunks)
- **Output**: JSON messages with audio responses

## When to use which:

**Use REST API when:**
- You have pre-recorded audio files
- You want simple request/response pattern
- You don't need real-time conversation
- You're building a batch processing system

**Use WebSocket when:**
- You want real-time voice chat
- You're recording audio from microphone
- You want continuous conversation flow
- You're building a voice assistant UI

## Input/Output Schemas:

### REST API Input:
```
POST /inference/voice-to-voice
- Audio file (multipart/form-data)
- Parameters: language, voice, model settings
```

### REST API Output:
```json
{
  "success": true,
  "data": {
    "transcription": "user's speech text",
    "response_text": "AI response text", 
    "audio_output_base64": "base64 audio data",
    "audio_format": "wav"
  }
}
```

### WebSocket Input:
```
Connect: ws://api.gmai.sa/ws/voice-conversation?input_language=auto&output_language=auto
Send: Binary audio data (WebM/WAV chunks)
```

### WebSocket Output:
```json
{
  "type": "voice_response",
  "success": true,
  "transcription": "user's speech",
  "response_text": "AI response", 
  "audio_base64": "base64 audio data",
  "processing_time": 2.5
}
```

## For UI Development:

**REST API**: Good for simple upload/download interface
**WebSocket**: Better for real-time chat interface with microphone recording

The main difference is REST = single interactions, WebSocket = continuous real-time conversation.Let me explain the difference between the WebSocket and Voice-to-Voice endpoints simply:

## They Are Different But Related

**Voice-to-Voice Endpoint (REST API)**:
- Traditional HTTP POST request
- Send audio file → Get audio file back
- One request = One response
- Good for: Simple voice conversion, batch processing

**WebSocket Voice Endpoint**:
- Real-time bidirectional connection
- Stay connected for continuous conversation
- Send audio → Get immediate responses
- Good for: Live chat, real-time conversation, voice assistants

## How to Use WebSocket Voice Endpoint

### Connection:
1. Connect to: `ws://api.gmai.sa/ws/voice-conversation`
2. Add query parameters: `?session_id=xyz&input_language=ar&output_language=ar&speaker_voice=female`

### Input Schema (What you send):
```
{
  "type": "audio_message",
  "audio_data": "base64_encoded_audio_bytes",
  "audio_format": "wav"
}
```

### Output Schema (What you receive):
```
{
  "type": "voice_response",
  "success": true,
  "transcription": "what user said in text",
  "response_text": "AI's text response", 
  "audio_base64": "base64_encoded_audio_response",
  "audio_format": "wav",
  "session_id": "xyz",
  "processing_time_ms": 1500
}
```

## UI Flow:
1. **Connect** to WebSocket with session parameters
2. **Record** user's voice → convert to base64
3. **Send** audio message via WebSocket
4. **Receive** response with transcription + AI audio
5. **Play** the AI's audio response
6. **Repeat** for continuous conversation (no need to reconnect)

## Key Difference:
- **REST**: One-shot (connect → send → receive → disconnect)
- **WebSocket**: Persistent (connect once → multiple send/receive → disconnect when done)

WebSocket is better for real-time voice chat because it maintains conversation context and is faster for multiple exchanges.