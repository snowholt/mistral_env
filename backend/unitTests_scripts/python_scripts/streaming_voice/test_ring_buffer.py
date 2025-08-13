import asyncio
import pytest
from beautyai_inference.services.voice.streaming.ring_buffer import PCMInt16RingBuffer

@pytest.mark.asyncio
async def test_ring_buffer_basic_write_read():
    rb = PCMInt16RingBuffer(sample_rate=16000, max_seconds=2.0)
    # 160 ms of audio (0.16 * 16000 = 2560 samples)
    import struct
    samples = [1000] * 2560
    data = struct.pack('<' + 'h'*len(samples), *samples)
    await rb.write(data)
    out = await rb.read_last_window(0.16)
    assert len(out) == len(data)
    # usage ratio should reflect proportion of capacity (~0.16/2.0)
    assert 0.05 < rb.usage_ratio() < 0.2

@pytest.mark.asyncio
async def test_ring_buffer_wrap_and_truncate():
    rb = PCMInt16RingBuffer(sample_rate=1000, max_seconds=1.0)  # small for test
    import struct
    # Write 1500 samples -> exceeds capacity (1000) triggers drop behavior
    samples = [i for i in range(1500)]
    data = struct.pack('<' + 'h'*len(samples), *samples)
    await rb.write(data)
    win = await rb.read_last_window(1.0)
    # Should only contain last 1000 samples (2 bytes each)
    assert len(win) == 1000 * 2
    # stats reflect a drop event
    assert rb.stats.total_dropped_events >= 1

@pytest.mark.asyncio
async def test_ring_buffer_rms():
    rb = PCMInt16RingBuffer(sample_rate=16000, max_seconds=1.0)
    import math, struct
    # 1000 samples of a sine wave amplitude 0.5
    N = 1000
    sine = [int(0.5 * 32767 * math.sin(2*math.pi*i/50)) for i in range(N)]
    data = struct.pack('<' + 'h'*N, *sine)
    await rb.write(data)
    rms = await rb.rms(1000/16000)
    assert 0.30 < rms < 0.38  # approximate
