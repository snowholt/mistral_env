import pytest
from beautyai_inference.services.voice.streaming.endpointing import (
    EndpointState, EndpointConfig, update_endpoint
)

FRAME_MS = 20

@pytest.fixture
def ep_state():
    cfg = EndpointConfig(frame_ms=FRAME_MS, calibration_ms=100, min_speech_ms=60, min_silence_ms=120)
    return EndpointState(config=cfg)


def advance(state, rms_values, tokens_list=None):
    events = []
    for i, rms in enumerate(rms_values):
        toks = None
        if tokens_list is not None:
            toks = tokens_list[i]
        events.extend(update_endpoint(state, rms, toks))
    return events


def test_endpoint_start_and_final(ep_state):
    # Calibration frames low RMS
    rms_sequence = [0.0001]*5  # 5*20=100ms calibration
    advance(ep_state, rms_sequence)
    assert ep_state.calibrated

    # Speech frames (above threshold) with tokens changing
    speech = [0.01]*6
    tokens = [[f"w{i}"] for i in range(6)]
    events = advance(ep_state, speech, tokens)
    # Should emit a start event once
    assert any(e.type == 'start' for e in events)

    # Silence with stable tokens to trigger finalize
    silence = [0.0001]*8  # 160 ms
    stable_tokens = [tokens[-1]]*8
    events = advance(ep_state, silence, stable_tokens)
    finals = [e for e in events if e.type == 'final']
    assert len(finals) == 1
    final = finals[0]
    assert final.reason in ('silence+stable', 'max_duration')


def test_endpoint_max_duration(ep_state):
    # Fast calibration
    advance(ep_state, [0.0001]*5)
    # Long voiced frames without silence
    voiced = [0.01]*((ep_state.config.max_utterance_ms // FRAME_MS) + 2)
    events = advance(ep_state, voiced, [["tok"]]*len(voiced))
    finals = [e for e in events if e.type == 'final']
    assert finals, 'Should finalize by max duration'
