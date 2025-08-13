"""Endpoint Detection Logic (Phase 3 – Mock Token Alignment)

Implements a lightweight adaptive RMS + token-diff based endpoint detector.
For Phase 3 we still rely on mock tokens (synthetic partials) but we wire
the state machine as it will be used when Whisper integration arrives.

Design Goals:
 - Pure, testable logic (no direct logging side effects).
 - State machine advanced by feeding new audio + (optional) token updates.
 - Adaptive threshold derived from initial calibration window.
 - Emits endpoint events when silence + token stability criteria met.

Future (Phase 4/5): integrate real token list, stability tail, max duration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EndpointConfig:
    sample_rate: int = 16000
    calibration_ms: int = 400
    min_speech_ms: int = 120  # need this much voiced energy to commit speech
    min_silence_ms: int = 600  # required trailing silence for endpoint
    max_utterance_ms: int = 12_000
    rms_factor: float = 1.8
    rms_margin: float = 0.0005
    frame_ms: int = 20  # assumed frame hop (client send cadence)


@dataclass
class EndpointState:
    config: EndpointConfig
    calibrated: bool = False
    baseline_rms: float = 0.0
    voiced_ms: int = 0
    silence_ms: int = 0
    utterance_ms: int = 0
    active: bool = False  # currently inside an utterance
    last_tokens: List[str] = field(default_factory=list)
    no_token_change_ms: int = 0
    pending_final: bool = False
    finalized: bool = False
    utterance_index: int = 0

    def reset_for_next(self) -> None:
        self.voiced_ms = 0
        self.silence_ms = 0
        self.utterance_ms = 0
        self.active = False
        self.last_tokens = []
        self.no_token_change_ms = 0
        self.pending_final = False
        self.finalized = False


@dataclass
class EndpointEvent:
    type: str  # 'start' | 'final'
    utterance_index: int
    reason: Optional[str] = None
    stable_tokens: Optional[int] = None
    voiced_ms: Optional[int] = None
    silence_ms: Optional[int] = None
    utterance_ms: Optional[int] = None


def update_endpoint(
    state: EndpointState,
    frame_rms: float,
    current_tokens: Optional[List[str]] = None,
) -> List[EndpointEvent]:
    """Advance endpoint state with one audio frame & optional token list.

    Returns list of EndpointEvent(s) generated this step (start, final).
    Token handling:
      - If tokens changed, reset no_token_change_ms.
      - If unchanged and inside utterance, accumulate no_token_change_ms.
    """
    cfg = state.config
    events: List[EndpointEvent] = []

    # Calibration phase: accumulate baseline mean RMS (simple exponential avg)
    if not state.calibrated:
        alpha = 0.15
        if state.baseline_rms == 0.0:
            state.baseline_rms = frame_rms
        else:
            state.baseline_rms = (1 - alpha) * state.baseline_rms + alpha * frame_rms
        if state.utterance_ms + cfg.frame_ms >= cfg.calibration_ms:
            state.calibrated = True
        state.utterance_ms += cfg.frame_ms
        return events  # no detection until calibrated

    threshold = state.baseline_rms * cfg.rms_factor + cfg.rms_margin
    voiced = frame_rms > threshold

    # Token stability tracking
    if current_tokens is not None:
        if current_tokens != state.last_tokens:
            state.no_token_change_ms = 0
            state.last_tokens = list(current_tokens)
        else:
            if state.active:
                state.no_token_change_ms += cfg.frame_ms

    # Inside utterance processing
    if state.active:
        state.utterance_ms += cfg.frame_ms
        if voiced:
            state.voiced_ms += cfg.frame_ms
            state.silence_ms = 0
        else:
            state.silence_ms += cfg.frame_ms

        # Endpoint conditions
        should_finalize = False
        reason = None
        if state.silence_ms >= cfg.min_silence_ms and state.no_token_change_ms >= 600:
            should_finalize = True
            reason = "silence+stable"
        elif state.utterance_ms >= cfg.max_utterance_ms:
            should_finalize = True
            reason = "max_duration"

        if should_finalize:
            events.append(EndpointEvent(
                type="final",
                utterance_index=state.utterance_index,
                reason=reason,
                stable_tokens=len(state.last_tokens),
                voiced_ms=state.voiced_ms,
                silence_ms=state.silence_ms,
                utterance_ms=state.utterance_ms,
            ))
            state.active = False
            state.finalized = True
            state.utterance_index += 1
            # Prepare for next after finalization; tokens reset next pass
            state.reset_for_next()
        return events

    # Not currently active; check for speech onset
    if voiced:
        state.voiced_ms += cfg.frame_ms
        if state.voiced_ms >= cfg.min_speech_ms:
            state.active = True
            events.append(EndpointEvent(type="start", utterance_index=state.utterance_index))
    else:
        # Decay voiced accumulation if intermittent noise
        state.voiced_ms = max(0, state.voiced_ms - cfg.frame_ms)

    return events


__all__ = [
    "EndpointConfig",
    "EndpointState",
    "EndpointEvent",
    "update_endpoint",
]
