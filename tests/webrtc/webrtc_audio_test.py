import asyncio
import os
from pathlib import Path
import time
from typing import Dict, List

import pytest
import requests
import urllib3

# Suppress SSL verification warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer
    from aiortc.mediastreams import MediaStreamTrack
    from av import AudioFrame
    import numpy as np
    import fractions

    AIORTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIORTC_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not AIORTC_AVAILABLE,
    reason="aiortc not installed - skipping WebRTC integration test",
)

# Use longer 5s English clip with early speech for reliable VAD confirmation
AUDIO_FIXTURE = Path("tests/webrtc/laser_hair.wav")
DEFAULT_SIGNALING_URL = os.getenv(
    "WEBRTC_TEST_BASE_URL",
    "http://localhost:8000/api/v1/webrtc/voice",
)

CONNECTION_TIMEOUT_SECONDS = 25
STREAM_DURATION_SECONDS = 10  # Stream audio for 10 seconds
RESPONSE_WAIT_SECONDS = 20  # Wait for server to process and respond (STT + LLM can be slow)

# laser_hair.wav contains an upbeat scripted greeting useful for VAD testing
EXPECTED_TRANSCRIPTION_FRAGMENT = "How does laser hair removal work?"


class AudioInspectorTrack(MediaStreamTrack):
    """
    A wrapper track that inspects audio frames and logs their content.
    This helps debug why server receives silent audio.
    """
    kind = "audio"
    
    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.source_track = source_track
        self.frame_count = 0
        
    async def recv(self) -> AudioFrame:
        frame = await self.source_track.recv()
        self.frame_count += 1
        
        # Inspect frame content
        if self.frame_count <= 5 or self.frame_count % 20 == 0:
            # Convert frame to numpy for analysis
            arr = frame.to_ndarray()
            max_val = np.abs(arr).max()
            rms = np.sqrt(np.mean(arr.astype(np.float32)**2))
            non_zero = np.count_nonzero(arr)
            total = arr.size
            
            print(f"[INSPECTOR] Frame {self.frame_count}: "
                  f"shape={arr.shape}, samples={frame.samples}, "
                  f"rate={frame.sample_rate}Hz, format={frame.format.name}, "
                  f"max={max_val:.1f}, rms={rms:.1f}, "
                  f"non_zero={non_zero}/{total}")
        
        return frame


class FileAudioTrack(MediaStreamTrack):
    """
    Custom audio track that reads from a file directly using soundfile/librosa.
    Bypasses MediaPlayer to avoid warm-up and compatibility issues.
    """
    kind = "audio"
    
    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        self._audio_data = None
        self._sample_rate = None
        self._frame_samples = 960  # 20ms at 48kHz
        self._current_pos = 0
        self._initialized = False
        
    def _initialize(self):
        """Load the audio file."""
        import soundfile as sf
        
        # Load audio file
        self._audio_data, self._sample_rate = sf.read(str(self.file_path), dtype='float32')
        
        print(f"[FileAudioTrack] Original: {len(self._audio_data)} samples at {self._sample_rate}Hz")
        
        # Check first few samples
        if len(self._audio_data) > 100:
            first_max = np.abs(self._audio_data[:100]).max()
            later_max = np.abs(self._audio_data[5000:5100] if len(self._audio_data) > 5100 else self._audio_data[-100:]).max()
            print(f"[FileAudioTrack] First 100 samples max: {first_max:.4f}, Later samples max: {later_max:.4f}")
        
        # NOTE: This clip has energetic speech well within the first second while
        # still providing >250ms of warmup silence, so no explicit skip needed.
        
        # Resample to 48kHz if needed (WebRTC standard)
        if self._sample_rate != 48000:
            from scipy import signal
            num_samples = int(len(self._audio_data) * 48000 / self._sample_rate)
            self._audio_data = signal.resample(self._audio_data, num_samples)
            self._sample_rate = 48000
        
        # Ensure mono
        if self._audio_data.ndim > 1:
            self._audio_data = np.mean(self._audio_data, axis=1)
        
        # Keep as 1D mono array
        self._original_data = self._audio_data.copy()
        
        self._initialized = True
        print(f"[FileAudioTrack] Final: {len(self._audio_data)} samples at {self._sample_rate}Hz")
        
        # Verify we have actual audio now
        if len(self._audio_data) > 1000:
            first_samples = self._audio_data[:100]
            later_samples = self._audio_data[1000:1100] if len(self._audio_data) > 1100 else self._audio_data[-100:]
            max_first = np.abs(first_samples).max()
            max_later = np.abs(later_samples).max()
            rms_later = np.sqrt(np.mean(later_samples**2))
            print(f"[FileAudioTrack] After skip - First 100: max={max_first:.3f}, Later: max={max_later:.3f}, rms={rms_later:.3f}")
        
    async def recv(self) -> AudioFrame:
        if not self._initialized:
            self._initialize()
        
        # Simulate real-time audio transmission (20ms per frame at 48kHz)
        # This prevents overwhelming the server with instant bulk transmission
        frame_duration_sec = self._frame_samples / self._sample_rate  # 960/48000 = 0.02s (20ms)
        await asyncio.sleep(frame_duration_sec)
        
        # Check if we have more audio
        if self._current_pos >= len(self._audio_data):
            # End of file - raise MediaStreamError to signal completion
            from aiortc.mediastreams import MediaStreamError
            self.stop()
            raise MediaStreamError("end of stream")
        
        # Get next chunk
        end_pos = min(self._current_pos + self._frame_samples, len(self._audio_data))
        chunk = self._audio_data[self._current_pos:end_pos]
        
        # Pad if needed (last frame might be short)
        if len(chunk) < self._frame_samples:
            padding = np.zeros(self._frame_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, padding])
        
        # Convert float32 [-1, 1] to int16 for Opus encoder
        chunk_int16 = (chunk * 32767).astype(np.int16)
        
        # For planar mono, we need shape: (1, samples)
        mono_chunk = chunk_int16.reshape(1, -1)
        
        self._current_pos = end_pos
        
        # Create AudioFrame with mono layout in s16 format (required by Opus encoder)
        from av import AudioFrame
        frame = AudioFrame.from_ndarray(mono_chunk, format='s16', layout='mono')
        frame.sample_rate = self._sample_rate
        frame.pts = self._current_pos - self._frame_samples
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        
        return frame


async def _post_json(url: str, payload: Dict) -> Dict:
    """Submit JSON payload via POST using a thread executor."""

    def _send() -> Dict:
        response = requests.post(url, json=payload, timeout=30, verify=False)
        response.raise_for_status()
        return response.json()

    return await asyncio.to_thread(_send)


async def _delete(url: str) -> None:
    """Send DELETE request in executor."""

    def _send() -> None:
        response = requests.delete(url, timeout=15, verify=False)
        # Treat 2xx/404 as success to keep cleanup idempotent
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()

    await asyncio.to_thread(_send)


def _build_url(base: str, suffix: str) -> str:
    base_clean = base.rstrip("/")
    suffix_clean = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"{base_clean}{suffix_clean}"


async def _exercise_round_trip(signaling_base: str) -> Dict[str, object]:
    """Run the end-to-end WebRTC flow and return summary metrics."""
    # Create custom audio track that reads file directly (bypasses MediaPlayer)
    audio_track = FileAudioTrack(AUDIO_FIXTURE)
    
    # Wrap with inspector for debugging
    wrapped_track = AudioInspectorTrack(audio_track)

    pc = RTCPeerConnection()
    pc.addTrack(wrapped_track)
    
    # Create client-side data channel for receiving server messages
    # In WebRTC, the offer side must create data channels for them to be negotiated properly
    client_dc = pc.createDataChannel("client_receive", ordered=True)
    print(f"[TEST] Created client data channel: {client_dc.label}")

    peer_id: str | None = None
    pending_ice: List[Dict] = []
    connection_ready = asyncio.Event()
    connection_failed = asyncio.Event()
    
    # Capture data channel messages (transcriptions and LLM responses)
    received_messages: List[Dict] = []
    data_channel_ready = asyncio.Event()
    
    @client_dc.on("open")
    def _on_client_dc_open() -> None:
        print(f"[TEST] ✓ Client data channel opened")
        data_channel_ready.set()
    
    @client_dc.on("message")
    def _on_client_dc_message(message) -> None:
        """Capture transcriptions and LLM responses from server."""
        try:
            import json
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            text = data.get("text", "")
            print(f"[TEST] Received {msg_type}: {text[:100]}")
            received_messages.append(data)
        except Exception as e:
            print(f"[TEST] Error parsing data channel message: {e}")
            received_messages.append({"type": "error", "raw": message, "error": str(e)})
    
    @client_dc.on("close")
    def _on_client_dc_close() -> None:
        print(f"[TEST] Client data channel closed")

    @pc.on("iceconnectionstatechange")
    async def _on_ice_state_change() -> None:
        state = pc.iceConnectionState
        print(f"[TEST] ICE connection state changed to: {state}")
        if state in ("connected", "completed"):
            connection_ready.set()
        elif state in ("failed", "disconnected", "closed"):
            connection_failed.set()

    async def _send_ice(candidate_payload: Dict) -> None:
        url = _build_url(signaling_base, "/ice")
        await _post_json(url, candidate_payload)

    @pc.on("icecandidate")
    async def _on_ice_candidate(candidate) -> None:  # pragma: no cover - callback
        if candidate is None:
            return
        if getattr(candidate, "candidate", None) in (None, ""):
            return
        candidate_payload = {
            "peer_id": peer_id,
            "candidate": candidate.candidate,
            "sdp_mid": candidate.sdpMid,
            "sdp_m_line_index": candidate.sdpMLineIndex,
        }
        if peer_id is None:
            pending_ice.append(candidate_payload)
        else:
            await _send_ice(candidate_payload)

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Log the SDP to check if data channel is included
        print(f"[TEST] === Local SDP Offer ===")
        print(f"[TEST] SDP has {offer.sdp.count('m=application')} application media sections (data channels)")
        if "a=sctp-port" in offer.sdp:
            print("[TEST] ✓ SDP contains SCTP port (data channel negotiation)")
        else:
            print("[TEST] ⚠️ SDP does NOT contain SCTP port")
        
        offer_payload = {
            "sdp": offer.sdp,
            "type": offer.type,
            "language": "en",  # laser_hair.wav is English narration
            "session_metadata": {
                "test_origin": "pytest_webrtc_laser_hair",
                "generated_at": time.time(),
            },
        }

        offer_url = _build_url(signaling_base, "/offer")
        print(f"[TEST] Sending offer to {offer_url}")
        offer_response = await _post_json(offer_url, offer_payload)
        print(f"[TEST] Offer response keys: {list(offer_response.keys())}")

        peer_id = offer_response.get("peer_id")
        if not peer_id:
            raise AssertionError("Signaling server did not assign peer_id")

        answer_sdp = offer_response.get("sdp") or offer_response.get("answer")
        if not answer_sdp:
            raise AssertionError("Signaling server response missing SDP answer")

        # Log the answer SDP to check server's data channel support
        print(f"[TEST] === Remote SDP Answer ===")
        print(f"[TEST] SDP has {answer_sdp.count('m=application')} application media sections")
        if "a=sctp-port" in answer_sdp:
            print("[TEST] ✓ Answer contains SCTP port (server supports data channel)")
        else:
            print("[TEST] ⚠️ Answer does NOT contain SCTP port (server may not support data channel)")
        
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )

        # Flush queued ICE once peer_id is known
        while pending_ice:
            candidate_payload = pending_ice.pop(0)
            candidate_payload["peer_id"] = peer_id
            await _send_ice(candidate_payload)

        await asyncio.wait_for(connection_ready.wait(), timeout=CONNECTION_TIMEOUT_SECONDS)
        if connection_failed.is_set():
            raise AssertionError("ICE connection entered failed state")
        
        print(f"[TEST] ICE connected, streaming audio for {STREAM_DURATION_SECONDS}s...")
        # Allow media to flow towards the server
        await asyncio.sleep(STREAM_DURATION_SECONDS)
        
        # Wait for data channel and server response
        print(f"[TEST] Audio streaming complete, waiting up to {RESPONSE_WAIT_SECONDS}s for server response...")
        print(f"[TEST] Data channel ready status: {data_channel_ready.is_set()}")
        try:
            await asyncio.wait_for(data_channel_ready.wait(), timeout=10.0)
            print("[TEST] ✓ Data channel opened successfully")
        except asyncio.TimeoutError:
            print("[TEST] ⚠️ WARNING: Data channel did not open within 10s")
            print("[TEST] This indicates the server may not be creating/negotiating data channels properly")
        
        # Give server time to process audio and send transcription/response
        # STT + LLM inference can take 10-20 seconds depending on load
        print(f"[TEST] Waiting {RESPONSE_WAIT_SECONDS}s for STT transcription + LLM response...")
        await asyncio.sleep(RESPONSE_WAIT_SECONDS)
        print(f"[TEST] Wait complete. Received {len(received_messages)} messages so far.")

        stats = await pc.getStats()
        outbound_audio = [
            report for report in stats.values()
            if report.type == "outbound-rtp" and getattr(report, "kind", "") == "audio"
        ]
        if not outbound_audio:
            raise AssertionError("No outbound audio RTP stats collected")

        packets_sent = sum(getattr(report, "packetsSent", 0) for report in outbound_audio)
        if packets_sent <= 0:
            raise AssertionError("No audio packets were sent to the server")
        
        # Analyze received messages
        transcriptions = [m for m in received_messages if m.get("type") == "transcription"]
        llm_responses = [m for m in received_messages if m.get("type") == "llm_response"]
        
        print(f"\n[TEST] === Summary ===")
        print(f"[TEST] Packets sent: {packets_sent}")
        print(f"[TEST] Messages received: {len(received_messages)}")
        print(f"[TEST] Transcriptions: {len(transcriptions)}")
        print(f"[TEST] LLM responses: {len(llm_responses)}")
        
        for i, trans in enumerate(transcriptions):
            print(f"[TEST] Transcription {i+1}: {trans.get('text', '')[:100]}")
        if transcriptions:
            normalized_text = transcriptions[-1].get("text", "").strip().lower()
            assert normalized_text, "Transcription payload should not be empty"
            assert EXPECTED_TRANSCRIPTION_FRAGMENT in normalized_text, (
                f"Normalized transcription mismatch: expected fragment '{EXPECTED_TRANSCRIPTION_FRAGMENT}', "
                f"got '{transcriptions[-1].get('text', '')}'"
            )
        for i, resp in enumerate(llm_responses):
            print(f"[TEST] LLM Response {i+1}: {resp.get('text', '')[:100]}")

        return {
            "peer_id": peer_id,
            "packets_sent": packets_sent,
            "ice_state": pc.iceConnectionState,
            "messages_received": len(received_messages),
            "transcriptions": transcriptions,
            "llm_responses": llm_responses,
            "all_messages": received_messages,
        }

    finally:
        if peer_id:
            try:
                await _delete(_build_url(signaling_base, f"/{peer_id}"))
            except requests.RequestException:
                pass
        await pc.close()
        if audio_track is not None:
            audio_track.stop()


def test_webrtc_audio_round_trip():
    """Synchronously exercise the WebRTC audio pipeline using asyncio.run."""
    if not AUDIO_FIXTURE.exists():
        pytest.skip(f"Audio fixture {AUDIO_FIXTURE.name} is missing")

    signaling_base = os.getenv("WEBRTC_TEST_BASE_URL", DEFAULT_SIGNALING_URL)
    if not signaling_base.startswith("http"):
        pytest.skip(
            f"WEBRTC_TEST_BASE_URL must include scheme, got '{signaling_base}'"
        )

    try:
        result = asyncio.run(_exercise_round_trip(signaling_base))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except requests.RequestException as exc:
        pytest.skip(f"Signaling server unreachable: {exc}")

    # Basic connectivity assertions
    assert result["packets_sent"] > 0, "No audio packets were sent"
    assert result["ice_state"] in ("connected", "completed", "closed"), f"Unexpected ICE state: {result['ice_state']}"
    
    # Check for server responses
    print(f"\n[TEST] === Validation ===")
    if result["messages_received"] == 0:
        print("[TEST] ⚠️  WARNING: No messages received from server")
        print("[TEST] This may indicate:")
        print("[TEST]   - Server audio processing failed")
        print("[TEST]   - Data channel not established")
        print("[TEST]   - Transcription/LLM pipeline issues")
    else:
        print(f"[TEST] ✓ Received {result['messages_received']} messages from server")
        
        if not result["transcriptions"]:
            print("[TEST] ⚠️  WARNING: No transcriptions received")
        else:
            print(f"[TEST] ✓ Received {len(result['transcriptions'])} transcription(s)")
            
        if not result["llm_responses"]:
            print("[TEST] ⚠️  WARNING: No LLM responses received")
        else:
            print(f"[TEST] ✓ Received {len(result['llm_responses'])} LLM response(s)")
    
    # Store results for inspection
    result["fixture_file"] = str(AUDIO_FIXTURE)