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
    min_speech_ms: int = 480  # need this much voiced energy to commit speech (increased from 120ms to reduce fragmentation)
    min_silence_ms: int = 720  # required trailing silence for endpoint (increased from 600ms for stability)
    token_stable_ms: int = 600  # how long tokens must remain unchanged to finalize
    max_utterance_ms: int = 12_000
    rms_factor: float = 1.8
    rms_margin: float = 0.0005
    frame_ms: int = 20  # assumed frame hop (client send cadence)
    # New hysteresis parameters for stable finalization
    min_token_growth_cycles: int = 3  # require token changes across this many cycles before allowing finalization
    stability_buffer_ms: int = 240  # additional buffer time after meeting basic criteria


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
    last_voiced_at_ms: int = 0  # timestamp (relative ms inside utterance) of last voiced frame
    # New hysteresis state
    token_growth_cycles: int = 0  # count of cycles with token changes
    total_cycles: int = 0  # total cycles since utterance start
    stability_met_at_ms: int = 0  # when basic criteria were first met
    min_criteria_met: bool = False  # basic silence+token criteria satisfied

    def reset_for_next(self) -> None:
        self.voiced_ms = 0
        self.silence_ms = 0
        self.utterance_ms = 0
        self.active = False
        self.last_tokens = []
        self.no_token_change_ms = 0
        self.pending_final = False
        self.finalized = False
        self.token_growth_cycles = 0
        self.total_cycles = 0
        self.stability_met_at_ms = 0
        self.min_criteria_met = False


@dataclass
class EndpointEvent:
    type: str  # 'start' | 'final'
    utterance_index: int
    reason: Optional[str] = None
    stable_tokens: Optional[int] = None
    voiced_ms: Optional[int] = None
    silence_ms: Optional[int] = None
    utterance_ms: Optional[int] = None
    end_silence_gap_ms: Optional[int] = None  # gap between last voiced frame and finalization


def update_endpoint(
    state: EndpointState,
    frame_rms: float,
    current_tokens: Optional[List[str]] = None,
) -> List[EndpointEvent]:
    """Advance endpoint state with one audio frame & optional token list.

    Returns list of EndpointEvent(s) generated this step (start, final).
    Token handling:
      - If tokens changed, reset no_token_change_ms and increment growth cycles.
      - If unchanged and inside utterance, accumulate no_token_change_ms.
    Hysteresis logic:
      - Require minimum token growth cycles before considering finalization.
      - Add stability buffer after basic criteria met to prevent premature finals.
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

    # Token stability tracking with growth counting
    if current_tokens is not None:
        if current_tokens != state.last_tokens:
            state.no_token_change_ms = 0
            state.last_tokens = list(current_tokens)
            if state.active:
                state.token_growth_cycles += 1
        else:
            if state.active:
                state.no_token_change_ms += cfg.frame_ms

    # Increment cycle counter for active utterances
    if state.active:
        state.total_cycles += 1

    # Token-driven activation: if we have decoded tokens but never crossed the
    # RMS gate to enter an utterance, promote to active state so downstream
    # logic (silence + token stability) can still finalize. This guards against
    # high initial baseline calibration (speech captured during calibration)
    # inflating the dynamic threshold and suppressing voiced detection, which
    # we observed prevented any 'start' / 'final' events while still producing
    # transcripts.
    if (not state.active) and current_tokens:
        state.active = True
        # Treat this frame as voiced activity; seed voiced_ms so we satisfy
        # min_speech_ms without waiting multiple large frames (e.g. 480ms).
        state.voiced_ms = max(state.voiced_ms, cfg.min_speech_ms)
        events.append(EndpointEvent(type="start", utterance_index=state.utterance_index))

    # Inside utterance processing
    if state.active:
        state.utterance_ms += cfg.frame_ms
        if voiced:
            state.voiced_ms += cfg.frame_ms
            state.silence_ms = 0
            state.last_voiced_at_ms = state.utterance_ms
        else:
            state.silence_ms += cfg.frame_ms

        # Check basic endpoint criteria
        basic_criteria_met = (
            state.silence_ms >= cfg.min_silence_ms and 
            state.no_token_change_ms >= cfg.token_stable_ms
        )
        
        # Advanced criteria with hysteresis
        token_growth_sufficient = state.token_growth_cycles >= cfg.min_token_growth_cycles
        
        # Track when basic criteria first met for stability buffer
        if basic_criteria_met and not state.min_criteria_met:
            state.min_criteria_met = True
            state.stability_met_at_ms = state.utterance_ms
        
        # Stability buffer check (if basic criteria met, wait additional buffer time)
        stability_buffer_satisfied = False
        if state.min_criteria_met:
            time_since_criteria_met = state.utterance_ms - state.stability_met_at_ms
            stability_buffer_satisfied = time_since_criteria_met >= cfg.stability_buffer_ms

        # Final endpoint decision
        should_finalize = False
        reason = None
        
        if state.utterance_ms >= cfg.max_utterance_ms:
            should_finalize = True
            reason = "max_duration"
        elif (basic_criteria_met and token_growth_sufficient and stability_buffer_satisfied):
            should_finalize = True
            reason = "silence+stable+growth"
        elif (basic_criteria_met and state.utterance_ms >= cfg.max_utterance_ms * 0.8):
            # Fallback: allow finalization without token growth if utterance is getting long
            should_finalize = True
            reason = "silence+stable+long"

        if should_finalize:
            end_gap = state.utterance_ms - state.last_voiced_at_ms if state.last_voiced_at_ms else state.silence_ms
            events.append(EndpointEvent(
                type="final",
                utterance_index=state.utterance_index,
                reason=reason,
                stable_tokens=len(state.last_tokens),
                voiced_ms=state.voiced_ms,
                silence_ms=state.silence_ms,
                utterance_ms=state.utterance_ms,
                end_silence_gap_ms=end_gap,
            ))
            state.active = False
            state.finalized = True
            state.utterance_index += 1
            # Prepare for next after finalization; tokens reset next pass
            state.reset_for_next()
            return events

    # Not currently active; check for speech onset
    if not state.active:
        if voiced:
            state.voiced_ms += cfg.frame_ms
            state.last_voiced_at_ms = state.utterance_ms
            if state.voiced_ms >= cfg.min_speech_ms:
                state.active = True
                events.append(EndpointEvent(type="start", utterance_index=state.utterance_index))
        else:
            # Decay voiced accumulation if intermittent noise
            state.voiced_ms = max(0, state.voiced_ms - cfg.frame_ms)
        
        # Always increment utterance_ms to track overall time
        state.utterance_ms += cfg.frame_ms

    return events


__all__ = [
    "EndpointConfig",
    "EndpointState",
    "EndpointEvent",
    "update_endpoint",
]
