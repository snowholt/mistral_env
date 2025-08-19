# Voice Streaming `/no_think` Fix

## Problem
The streaming voice websocket (`https://dev.gmai.sa/debug/pcm-upload`) was generating nonsensical responses to Arabic users because the LLM was running in thinking mode, which adds verbose reasoning that interferes with direct voice responses.

## Root Cause
Unlike the simple voice service which automatically appends `/no_think` to transcriptions, the streaming voice websocket was passing raw transcription text directly to the LLM without disabling thinking mode.

## Solution
Modified `/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/streaming_voice.py` to:

1. **Append `/no_think` to final transcripts** before passing them to the LLM pipeline
2. **Handle both direct processing and queued processing** paths 
3. **Add fallback for empty transcripts** (`"unclear audio /no_think"`)

### Changes Made

#### Primary Processing Path (Lines ~634-645)
```python
# Add /no_think to disable thinking mode for voice conversations
final_text_with_no_think = final_text.strip() + " /no_think" if final_text.strip() else "unclear audio /no_think"

state.llm_tts_task = asyncio.create_task(
    _process_final_transcript(
        utterance_index,
        final_text_with_no_think,
        language,
        state,
    )
)
```

#### Queued Processing Path (Lines ~648-651)
```python
# Queue this final for later processing (also add /no_think)
final_text_with_no_think = final_text.strip() + " /no_think" if final_text.strip() else "unclear audio /no_think"
state.pending_finals.append((utterance_index, final_text_with_no_think))
```

#### Documentation Update
Added note to `_process_final_transcript` function documenting that text should include `/no_think` suffix.

## Expected Impact
- **Faster responses**: Eliminates thinking mode overhead in voice conversations
- **Cleaner responses**: No verbose reasoning in TTS output
- **Consistent behavior**: Aligns streaming voice with simple voice service behavior
- **Better Arabic support**: Direct responses without thinking artifacts

## Testing
- Syntax validation passed
- Consistent with existing simple voice service implementation
- No breaking changes to API interface

## Files Modified
- `/home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/streaming_voice.py`

## Related
- Simple voice service implementation: `/home/lumi/beautyai/backend/src/beautyai_inference/services/voice/conversation/simple_voice_service.py` (lines 453-459)
- Issue URL: `https://dev.gmai.sa/debug/pcm-upload`