import asyncio
import websockets
import json
import base64
import pathlib

WS_URL = "ws://localhost:8000/api/v1/ws/simple-voice-chat?language=ar&voice_type=female"

async def main():
    print(f"Connecting to {WS_URL}")
    async with websockets.connect(WS_URL, max_size=50 * 1024 * 1024) as ws:
        # Receive welcome
        msg = await ws.recv()
        print("<--", msg)

        # Load small test webm if exists else send silence wav header fragments
        test_dir = pathlib.Path("voice_tests")
        sample = None
        for cand in ["test_webm_output.webm", "test_english_webm_output.webm"]:
            if (test_dir / cand).exists():
                sample = (test_dir / cand).read_bytes()
                break
        if sample is None:
            print("No sample webm found; sending empty bytes to test error handling")
            sample = b"\x1a\x45\xdf\xa3" * 200  # minimal EBML repeating

        # Simulate chunked MediaRecorder stream (split into 8KB chunks)
        chunk_size = 8192
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]
        print(f"Sending {len(chunks)} chunks totaling {len(sample)} bytes")
        for i, c in enumerate(chunks[:40]):  # limit
            await ws.send(c)
            await asyncio.sleep(0.03)  # 30ms pacing
            # Non-blocking receive of any server messages
            try:
                while True:
                    ws_msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print("<--", ws_msg[:160], "...")
            except asyncio.TimeoutError:
                pass

        # Allow processing
        await asyncio.sleep(5)
        # Drain any remaining messages
        try:
            while True:
                ws_msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print("<--", ws_msg[:200], "...")
        except asyncio.TimeoutError:
            pass

        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
