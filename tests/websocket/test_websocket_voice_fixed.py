#!/usr/bin/env python3
"""
Test WebSocket Voice Chat with fixed parameter handling
"""
import asyncio
import websockets
import json
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_voice():
    """Test the WebSocket voice chat endpoint with various parameter configurations"""
    
    # Test configurations
    test_configs = [
        {
            "name": "minimal_params",
            "config": {}  # Should use defaults/presets
        },
        {
            "name": "with_preset",
            "config": {"preset": "conversational"}
        },
        {
            "name": "with_temperature",
            "config": {"temperature": 0.7}
        },
        {
            "name": "full_params",
            "config": {
                "temperature": 0.8,
                "max_tokens": 150,
                "preset": "conversational"
            }
        }
    ]
    
    # Sample audio data (base64 encoded silence - for testing parameter handling)
    sample_audio = base64.b64encode(b'\x00' * 1024).decode('utf-8')
    
    for test_config in test_configs:
        logger.info(f"\n=== Testing {test_config['name']} ===")
        
        try:
            # Construct WebSocket URL
            ws_url = "ws://localhost:8000/ws/voice"
            if test_config['config']:
                # Add query parameters
                params = []
                for key, value in test_config['config'].items():
                    params.append(f"{key}={value}")
                if params:
                    ws_url += "?" + "&".join(params)
            
            logger.info(f"Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ WebSocket connection established")
                
                # Send test message
                test_message = {
                    "audio_data": sample_audio,
                    "session_id": f"test_{test_config['name']}",
                    "input_language": "en",
                    "output_language": "en"
                }
                
                logger.info("📤 Sending test message...")
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                logger.info("⏳ Waiting for response...")
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                
                # Parse response
                response_data = json.loads(response)
                logger.info(f"📥 Response received: {response_data.get('success', False)}")
                
                if response_data.get('success'):
                    logger.info("✅ Success! Key fields present:")
                    for key in ['response_text', 'audio_output_path', 'transcription']:
                        if key in response_data:
                            logger.info(f"  - {key}: present")
                        else:
                            logger.warning(f"  - {key}: MISSING")
                else:
                    logger.error(f"❌ Error: {response_data.get('error', 'Unknown error')}")
                    if 'details' in response_data:
                        logger.error(f"Details: {response_data['details']}")
                
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout waiting for response in {test_config['name']}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"❌ WebSocket connection closed in {test_config['name']}: {e}")
        except Exception as e:
            logger.error(f"❌ Error in {test_config['name']}: {e}")
        
        # Brief pause between tests
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_websocket_voice())
