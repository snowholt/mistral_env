#!/usr/bin/env python3
"""
BeautyAI Assistant Web UI
========================

A beautiful, user-friendly web interface for chatting with BeautyAI models.
Features animated 3D fractal background and comprehensive chat controls.
"""

import sys
import os
import logging
import asyncio
import uuid
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

try:
    from flask import Flask, render_template, request, jsonify, session, Response, send_file
    import aiohttp
    import json
    from datetime import datetime

except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install flask aiohttp")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'beautyai-secret-key-change-in-production'

# Configuration
BEAUTYAI_API_URL = os.environ.get('BEAUTYAI_API_URL', "http://localhost:8000")
CHAT_ENDPOINT = f"{BEAUTYAI_API_URL}/inference/chat"
MODELS_ENDPOINT = f"{BEAUTYAI_API_URL}/models"

class BeautyAIChatService:
    """Service for handling chat interactions with BeautyAI API."""
    
    def __init__(self, api_url: str = BEAUTYAI_API_URL):
        self.api_url = api_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_available_models(self):
        """Get list of available models with their status."""
        try:
            # Get all available models
            async with self.session.get(f"{self.api_url}/models/") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    # Get loaded models status
                    loaded_models = {}
                    try:
                        async with self.session.get(f"{self.api_url}/models/loaded") as loaded_response:
                            if loaded_response.status == 200:
                                loaded_data = await loaded_response.json()
                                loaded_list = loaded_data.get('data', {}).get('models', [])
                                for loaded_model in loaded_list:
                                    loaded_models[loaded_model.get('name')] = True
                    except Exception as e:
                        logger.warning(f"Could not fetch loaded models: {e}")
                    
                    # Combine model info with loaded status
                    model_details = []
                    for model in models:
                        model_name = model.get('name') or model.get('model_name') or str(model)
                        is_loaded = model_name in loaded_models
                        model_details.append({
                            'name': model_name,
                            'model_name': model_name,
                            'status': 'loaded' if is_loaded else 'unloaded',
                            'loaded': is_loaded,
                            'memory_usage_mb': 0,  # We could get this from system status if needed
                            'config': model if isinstance(model, dict) else {},
                            'is_default': model.get('is_default', False)
                        })
                    
                    return model_details
                return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    async def get_model_status(self, model_name: str):
        """Get status of a specific model."""
        try:
            async with self.session.get(f"{self.api_url}/models/{model_name}/status") as response:
                if response.status == 200:
                    return await response.json()
                return {'status': 'unknown', 'loaded': False}
        except Exception as e:
            logger.error(f"Error getting model status for {model_name}: {e}")
            return {'status': 'error', 'loaded': False}
    
    async def load_model(self, model_name: str):
        """Load a model for inference."""
        try:
            payload = {"force_reload": False}
            async with self.session.post(
                f"{self.api_url}/models/{model_name}/load",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"Load Error {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }

    async def unload_model(self, model_name: str):
        """Unload a model from memory."""
        try:
            async with self.session.post(
                f"{self.api_url}/models/{model_name}/unload",
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"Unload Error {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }

    async def send_chat_message(self, message_data: dict):
        """Send chat message to BeautyAI API."""
        try:
            # Ensure the message_data matches the expected API format exactly
            formatted_message = {
                "model_name": message_data.get("model_name"),
                "message": message_data.get("message"),
                "session_id": message_data.get("session_id"),
                "chat_history": message_data.get("chat_history", []),
                "generation_config": message_data.get("generation_config", {}),
                "stream": message_data.get("stream", False)
            }
            
            # Add optional fields if present
            if "disable_content_filter" in message_data:
                formatted_message["disable_content_filter"] = message_data["disable_content_filter"]
            if "content_filter_strictness" in message_data:
                formatted_message["content_filter_strictness"] = message_data["content_filter_strictness"]
            
            async with self.session.post(
                f"{self.api_url}/inference/chat",
                json=formatted_message,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    return {
                        'success': False,
                        'error': f"API Error {response.status}: {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }

    async def send_audio_chat_message(self, audio_file, form_data: dict):
        """Send audio file to BeautyAI audio-chat API."""
        try:
            # Prepare form data for multipart/form-data request
            data = aiohttp.FormData()
            
            # Add the audio file with proper content type detection
            content_type = 'audio/wav'  # Default
            filename = 'audio.wav'  # Default
            
            # Try to determine content type from form data or file extension
            if isinstance(audio_file, bytes):
                # For recorded audio, use default or detect from magic bytes
                if audio_file.startswith(b'RIFF') and b'WAVE' in audio_file[:20]:
                    content_type = 'audio/wav'
                    filename = 'recording.wav'
                elif audio_file.startswith(b'\x1a\x45\xdf\xa3'):  # WebM magic bytes
                    content_type = 'audio/webm'
                    filename = 'recording.webm'
                elif audio_file.startswith(b'OggS'):
                    content_type = 'audio/ogg'
                    filename = 'recording.ogg'
                elif audio_file.startswith(b'ID3') or audio_file.startswith(b'\xff\xfb'):
                    content_type = 'audio/mpeg'
                    filename = 'recording.mp3'
            
            logger.info(f"Sending audio file: {filename} ({content_type}), size: {len(audio_file)} bytes")
            data.add_field('audio_file', audio_file, filename=filename, content_type=content_type)
            
            # Check if thinking mode is disabled and add \no_think prefix instruction
            thinking_mode = form_data.get('thinking_mode', True)
            if not thinking_mode:
                logger.info("Thinking mode disabled, adding \\no_think prefix instruction for audio message")
                data.add_field('add_no_think_prefix', 'true')
            
            # Add all form parameters
            for key, value in form_data.items():
                if value is not None:
                    data.add_field(key, str(value))
            
            logger.info(f"Sending audio chat request to {self.api_url}/inference/audio-chat")
            logger.info(f"Form data keys: {list(form_data.keys())}")
            
            async with self.session.post(
                f"{self.api_url}/inference/audio-chat",
                data=data
            ) as response:
                response_text = await response.text()
                logger.info(f"Audio API response status: {response.status}")
                logger.debug(f"Audio API response body: {response_text[:500]}...")
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        return {
                            'success': False,
                            'error': f"Invalid JSON response from API: {str(e)}"
                        }
                else:
                    logger.error(f"Audio API Error {response.status}: {response_text}")
                    
                    # Try to parse error as JSON
                    try:
                        error_data = json.loads(response_text)
                        return {
                            'success': False,
                            'error': error_data.get('error', f"API Error {response.status}"),
                            'transcription_error': error_data.get('transcription_error'),
                            'chat_error': error_data.get('chat_error')
                        }
                    except json.JSONDecodeError:
                        return {
                            'success': False,
                            'error': f"Audio API Error {response.status}: {response_text}"
                        }
        except Exception as e:
            logger.error(f"Error sending audio chat message: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }

    async def send_voice_to_voice_message(self, audio_file, form_data: dict):
        """Send audio file to BeautyAI voice-to-voice API."""
        try:
            # Prepare form data for multipart/form-data request
            data = aiohttp.FormData()
            
            # Add the audio file with proper content type detection
            content_type = 'audio/wav'  # Default
            filename = 'audio.wav'  # Default
            
            # Try to determine content type from form data or file extension
            if isinstance(audio_file, bytes):
                # For recorded audio, use default or detect from magic bytes
                if audio_file.startswith(b'RIFF') and b'WAVE' in audio_file[:20]:
                    content_type = 'audio/wav'
                    filename = 'recording.wav'
                elif audio_file.startswith(b'\x1a\x45\xdf\xa3'):  # WebM magic bytes
                    content_type = 'audio/webm'
                    filename = 'recording.webm'
                elif audio_file.startswith(b'OggS'):
                    content_type = 'audio/ogg'
                    filename = 'recording.ogg'
                elif audio_file.startswith(b'ID3') or audio_file.startswith(b'\xff\xfb'):
                    content_type = 'audio/mpeg'
                    filename = 'recording.mp3'
            
            logger.info(f"Sending voice-to-voice audio file: {filename} ({content_type}), size: {len(audio_file)} bytes")
            data.add_field('audio_file', audio_file, filename=filename, content_type=content_type)
            
            # Add all form parameters
            for key, value in form_data.items():
                if value is not None:
                    data.add_field(key, str(value))
            
            logger.info(f"Sending voice-to-voice request to {self.api_url}/inference/voice-to-voice")
            logger.info(f"Form data keys: {list(form_data.keys())}")
            
            async with self.session.post(
                f"{self.api_url}/inference/voice-to-voice",
                data=data
            ) as response:
                response_text = await response.text()
                logger.info(f"Voice-to-voice API response status: {response.status}")
                logger.debug(f"Voice-to-voice API response body: {response_text[:500]}...")
                
                if response.status == 200:
                    try:
                        result = json.loads(response_text)
                        
                        # If the response includes audio data, handle it specially
                        if 'audio_data' in result:
                            # The audio data should already be in the response
                            return result
                        else:
                            # For responses without embedded audio, check if there's a file response
                            content_type = response.headers.get('content-type', '')
                            if content_type.startswith('audio/'):
                                # This is an audio file response
                                audio_data = await response.read()
                                return {
                                    'success': True,
                                    'audio_data': audio_data,
                                    'audio_output_format': content_type.split('/')[-1],
                                    'audio_size_bytes': len(audio_data)
                                }
                            else:
                                return result
                                
                    except json.JSONDecodeError as e:
                        # If it's not JSON, it might be raw audio data
                        content_type = response.headers.get('content-type', '')
                        if content_type.startswith('audio/'):
                            audio_data = await response.read()
                            return {
                                'success': True,
                                'audio_data': audio_data,
                                'audio_output_format': content_type.split('/')[-1],
                                'audio_size_bytes': len(audio_data)
                            }
                        else:
                            logger.error(f"Failed to parse voice-to-voice JSON response: {e}")
                            return {
                                'success': False,
                                'error': f"Invalid JSON response from API: {str(e)}"
                            }
                else:
                    logger.error(f"Voice-to-voice API Error {response.status}: {response_text}")
                    
                    # Try to parse error as JSON
                    try:
                        error_data = json.loads(response_text)
                        return {
                            'success': False,
                            'error': error_data.get('error', f"API Error {response.status}"),
                            'transcription_error': error_data.get('transcription_error'),
                            'generation_error': error_data.get('generation_error'),
                            'tts_error': error_data.get('tts_error'),
                            'errors': error_data.get('errors', [])
                        }
                    except json.JSONDecodeError:
                        return {
                            'success': False,
                            'error': f"Voice-to-voice API Error {response.status}: {response_text}"
                        }
        except Exception as e:
            logger.error(f"Error sending voice-to-voice message: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }

# Global service instance
chat_service = BeautyAIChatService()

@app.route('/')
def index():
    """Modern chat interface with both text and voice support."""
    # Initialize session if needed
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('chat_ui.html')

@app.route('/legacy')
def legacy():
    """Legacy chat interface."""
    # Initialize session if needed
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('chat_ui.html')

@app.route('/ws/simple-voice')
def simple_voice_websocket():
    """WebSocket endpoint for simple voice-to-voice chat."""
    # For now, provide info about the backend WebSocket
    backend_ws_url = "ws://localhost:8000/ws/simple-voice"
    return jsonify({
        "message": "WebSocket endpoint available at backend",
        "websocket_url": backend_ws_url,
        "note": "Connect directly to backend WebSocket for real-time voice chat"
    })

@app.route('/debug')
def debug():
    """Debug page for thinking mode issue."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>🔍 Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .test-checkbox { margin: 20px 0; }
        #results { margin-top: 20px; }
        .iframe-container { margin: 20px 0; }
        iframe { width: 100%; height: 400px; border: 1px solid #ccc; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Real-Time Debugging</h2>
        <p>This page will help us debug the thinking mode checkbox issue.</p>
        
        <div class="error">
            <strong>Current Issue:</strong> The checkbox appears to change but backend always receives <code>thinking_enabled: true</code>
        </div>
    </div>
    
    <div class="debug-panel">
        <h2>Test Checkbox</h2>
        <div class="test-checkbox">
            <label>
                <input type="checkbox" id="testCheckbox" checked> Test Thinking Mode (simulates the real checkbox)
            </label>
        </div>
        
        <button onclick="debugCheckboxState()">Check Checkbox State</button>
        <button onclick="testPayloadGeneration()">Test Payload Generation</button>
        <button onclick="testAPICall()">Test Real API Call</button>
        <button onclick="inspectMainApp()">Inspect Main App</button>
        
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Live Web UI (for comparison)</h2>
        <div class="iframe-container">
            <iframe src="/" id="webui-frame"></iframe>
        </div>
        <button onclick="inspectIframeCheckbox()">Inspect Web UI Checkbox</button>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Console Commands</h2>
        <p>Run these in the main Web UI console (open in new tab):</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check app instance  
const app = window.beautyAIChat;
console.log('App instance:', app);
console.log('Parameter controls:', app?.parameterControls?.enable_thinking?.checked);

// 3. Force checkbox change
if (checkbox) {
    checkbox.checked = false;
    checkbox.dispatchEvent(new Event('change'));
    console.log('Forced to false, now:', checkbox.checked);
}

// 4. Monitor all clicks
document.addEventListener('click', (e) => {
    if (e.target.id === 'enable_thinking') {
        console.log('Thinking checkbox clicked, new state:', e.target.checked);
    }
});

// 5. Test payload creation manually
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false,
        message: "test"
    };
    console.log('Manual payload test:', payload);
    return payload;
}</div>
    </div>
    
    <script>
        function debugCheckboxState() {
            const checkbox = document.getElementById('testCheckbox');
            const results = document.getElementById('results');
            
            const info = {
                exists: !!checkbox,
                checked: checkbox?.checked,
                value: checkbox?.value,
                type: checkbox?.type,
                disabled: checkbox?.disabled,
                id: checkbox?.id
            };
            
            results.innerHTML = `
                <div class="status info">
                    <h3>Local Test Checkbox State:</h3>
                    <div class="code">${JSON.stringify(info, null, 2)}</div>
                    <p><strong>This checkbox is:</strong> ${checkbox?.checked ? 'CHECKED' : 'UNCHECKED'}</p>
                </div>
            `;
        }
        
        function testPayloadGeneration() {
            const checkbox = document.getElementById('testCheckbox');
            const results = document.getElementById('results');
            
            // Simulate the exact logic from main.js
            const payload = {
                thinking_mode: checkbox?.checked || false,
                message: "test message",
                model_name: "qwen3-unsloth-q4ks"
            };
            
            results.innerHTML = `
                <div class="status ${checkbox?.checked ? 'error' : 'success'}">
                    <h3>Simulated Payload Generation:</h3>
                    <div class="code">${JSON.stringify(payload, null, 2)}</div>
                    <p><strong>Expected:</strong> thinking_mode should be ${checkbox?.checked}</p>
                    <p><strong>Logic:</strong> checkbox?.checked || false = ${checkbox?.checked || false}</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const checkbox = document.getElementById('testCheckbox');
            const results = document.getElementById('results');
            
            const payload = {
                thinking_mode: checkbox?.checked || false,
                message: "debug test from debug page",
                model_name: "qwen3-unsloth-q4ks",
                session_id: "debug_session",
                chat_history: [],
                generation_config: {
                    max_tokens: 50,
                    temperature: 0.3,
                    top_p: 0.95,
                    top_k: 20,
                    repetition_penalty: 1.1,
                    min_p: 0.05
                },
                disable_content_filter: true,
                content_filter_strictness: "disabled"
            };
            
            results.innerHTML = `
                <div class="status info">
                    <h3>Testing Real API Call...</h3>
                    <div class="code">Sending payload: ${JSON.stringify(payload, null, 2)}</div>
                </div>
            `;
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                const isCorrect = data.thinking_enabled === payload.thinking_mode;
                results.innerHTML = `
                    <div class="status ${isCorrect ? 'success' : 'error'}">
                        <h3>API Response:</h3>
                        <div class="code">Sent: thinking_mode = ${payload.thinking_mode}
Received: thinking_enabled = ${data.thinking_enabled}
Has thinking content: ${!!data.thinking_content}

Full response: ${JSON.stringify(data, null, 2)}</div>
                        <p><strong>Result:</strong> ${isCorrect ? '✅ WORKING CORRECTLY' : '❌ BUG CONFIRMED'}</p>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `
                    <div class="status error">
                        <h3>API Error:</h3>
                        <div class="code">${error.message}</div>
                    </div>
                `;
            });
        }
        
        function inspectMainApp() {
            const results = document.getElementById('results');
            results.innerHTML = `
                <div class="status info">
                    <h3>Main App Inspection:</h3>
                    <p>Open the main Web UI in a new tab and run these commands in the console:</p>
                    <div class="code">// Check if app exists
console.log('App:', window.beautyAIChat);

// Check parameter controls
console.log('Controls:', window.beautyAIChat?.parameterControls);

// Check thinking checkbox specifically
const thinkingCheckbox = window.beautyAIChat?.parameterControls?.enable_thinking;
console.log('Thinking checkbox:', thinkingCheckbox);
console.log('Thinking checked:', thinkingCheckbox?.checked);

// Check DOM element directly
const domCheckbox = document.getElementById('enable_thinking');
console.log('DOM checkbox:', domCheckbox);
console.log('DOM checked:', domCheckbox?.checked);

// Compare them
console.log('Are they the same element?', thinkingCheckbox === domCheckbox);</div>
                    <p><a href="/" target="_blank">Open Main Web UI in New Tab</a></p>
                </div>
            `;
        }
        
        function inspectIframeCheckbox() {
            const iframe = document.getElementById('webui-frame');
            const results = document.getElementById('results');
            
            try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                const checkbox = iframeDoc.getElementById('enable_thinking');
                
                if (checkbox) {
                    const info = {
                        exists: true,
                        checked: checkbox.checked,
                        value: checkbox.value,
                        disabled: checkbox.disabled
                    };
                    
                    results.innerHTML = `
                        <div class="status info">
                            <h3>Web UI Checkbox State (via iframe):</h3>
                            <div class="code">${JSON.stringify(info, null, 2)}</div>
                            <button onclick="toggleIframeCheckbox()">Toggle Iframe Checkbox</button>
                        </div>
                    `;
                } else {
                    results.innerHTML = `
                        <div class="status error">
                            <h3>Cannot find checkbox in iframe</h3>
                            <p>This might be due to CORS restrictions or the page not being fully loaded.</p>
                        </div>
                    `;
                }
            } catch (e) {
                results.innerHTML = `
                    <div class="status error">
                        <h3>Cannot access iframe content:</h3>
                        <div class="code">${e.message}</div>
                        <p>This is normal due to browser security. Use the manual inspection instead.</p>
                    </div>
                `;
            }
        }
        
        function toggleIframeCheckbox() {
            const iframe = document.getElementById('webui-frame');
            try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                const checkbox = iframeDoc.getElementById('enable_thinking');
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change'));
                    inspectIframeCheckbox(); // Refresh display
                }
            } catch (e) {
                console.error('Cannot toggle iframe checkbox:', e);
            }
        }
        
        // Auto-run initial test
        window.onload = function() {
            debugCheckboxState();
        };
        
        // Monitor local test checkbox
        document.getElementById('testCheckbox').addEventListener('change', function(e) {
            console.log('Test checkbox changed to:', e.target.checked);
            debugCheckboxState();
        });
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        # Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data:
            # Handle both string and boolean values
            if isinstance(data['thinking_mode'], str):
                # Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            else:
                thinking_mode = bool(data['thinking_mode'])
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        elif 'enable_thinking' in data:
            # Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        else:
            logger.info("No thinking_mode parameter found in request")
        
        # Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message:
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        # Add content filter control
        if data.get('disable_content_filter', False):
            message_data['disable_content_filter'] = True
        
        if 'content_filter_strictness' in data:
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        
        # Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None:
            message_data['thinking_mode'] = thinking_mode
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False):
            # Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        # Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data:
            # Handle both string and boolean values
            if isinstance(data['thinking_mode'], str):
                # Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            else:
                thinking_mode = bool(data['thinking_mode'])
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        elif 'enable_thinking' in data:
            # Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        else:
            logger.info("No thinking_mode parameter found in request")
        
        # Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message:
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        # Add content filter control
        if data.get('disable_content_filter', False):
            message_data['disable_content_filter'] = True
        
        if 'content_filter_strictness' in data:
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        
        # Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None:
            message_data['thinking_mode'] = thinking_mode
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False):
            # Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
        .iframe-container { margin: 20px 0; }
        iframe { width: 100%; height: 400px; border: 1px solid #ccc; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        elif engine_type == 'transformers':
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result === test.expected;
                console.log(`Test: checked=${test.checked}, result=${result}, expected=${test.expected}, passed=${passed}`);
                if (!passed) allPassed = false;
            });
            
            results.innerHTML = `
                <div class="status ${allPassed ? 'success' : 'error'}">
                    <h3>Logic Test: ${allPassed ? 'PASSED' : 'FAILED'}</h3>
                    <p>Check console for details</p>
                </div>
            `;
        }
        
        function testAPICall() {
            const results = document.getElementById('results');
            
            // Test API call with thinking disabled
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: 'qwen3-unsloth-q4ks',
                    message: 'test thinking disabled',
                    thinking_mode: false,
                    disable_content_filter: true
                })
            })
            .then(response => response.json())
            .then(data => {
                const success = data.thinking_enabled === false;
                results.innerHTML = `
                    <div class="status ${success ? 'success' : 'error'}">
                        <h3>API Test: ${success ? 'PASSED' : 'FAILED'}</h3>
                        <p><strong>Sent:</strong> thinking_mode: false</p>
                        <p><strong>Received:</strong> thinking_enabled: ${data.thinking_enabled}</p>
                        <p><strong>Has thinking content:</strong> ${!!data.thinking_content}</p>
                        <div class="code">${JSON.stringify(data, null, 2)}</div>
                    </div>
                `;
            })
            .catch(error => {
                results.innerHTML = `<div class="status error">API Error: ${error.message}</div>`;
            });
        }
        
        function goToMainUI() {
            window.open('/', '_blank');
        }
    </script>
</body>
</html>
    '''

@app.route('/api/models')
def get_models():
    """Get available models endpoint."""
    async def _get_models():
        async with BeautyAIChatService() as service:
            models = await service.get_available_models()
            return models
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        models = loop.run_until_complete(_get_models())
        loop.close()
        
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load a model endpoint."""
    async def _load_model():
        async with BeautyAIChatService() as service:
            return await service.load_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} loaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error loading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Load model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name):
    """Unload a model endpoint."""
    async def _unload_model():
        async with BeautyAIChatService() as service:
            return await service.unload_model(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_unload_model())
        loop.close()
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Model {model_name} unloaded successfully',
                'model_name': model_name
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error unloading model')
            }), 500
            
    except Exception as e:
        logger.error(f"Unload model error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<model_name>/status')
def get_model_status(model_name):
    """Get model status endpoint."""
    async def _get_status():
        async with BeautyAIChatService() as service:
            return await service.get_model_status(model_name)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_get_status())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result
        })
            
    except Exception as e:
        logger.error(f"Get model status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try {
        data = request.get_json()
        
        # Get model info to determine engine type
        model_name = data.get('model_name', 'qwen3-unsloth-q4ks')
        
        # Check model engine type for parameter compatibility
        async def _get_model_info():
            async with BeautyAIChatService() as service:
                models = await service.get_available_models()
                return next((m for m in models if (m.get('name') or m.get('model_name')) == model_name), None)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        model_info = loop.run_until_complete(_get_model_info())
        loop.close()
        
        engine_type = None
        if model_info and model_info.get('config'):
            engine_type = model_info['config'].get('engine_type')
        
        # Prepare base generation config
        generation_config = {
            'max_tokens': int(data.get('max_new_tokens', data.get('max_tokens', 2048))),
            'temperature': float(data.get('temperature', 0.3)),
            'top_p': float(data.get('top_p', 0.95)),
            'repetition_penalty': float(data.get('repetition_penalty', 1.1))
        }
        
        # Add engine-specific parameters
        if engine_type == 'llama.cpp':
            # LlamaCpp supports more parameters
            if 'top_k' in data:
                generation_config['top_k'] = int(data['top_k'])
            if 'min_p' in data:
                generation_config['min_p'] = float(data['min_p'])
        } else if engine_type == 'transformers' {
            # Transformers engine is more limited, stick to basic parameters
            logger.info(f"Using Transformers engine for {model_name}, using basic parameters only")
        }
        
        # Prepare chat message with correct API structure
        original_message = data.get('message', '')
        
        // Check if thinking mode is disabled and add \no_think prefix
        thinking_mode = None
        if 'thinking_mode' in data {
            // Handle both string and boolean values
            if isinstance(data['thinking_mode'], str) {
                // Handle string values: "enable"/"disable" or "true"/"false"
                thinking_value = data['thinking_mode'].lower()
                thinking_mode = thinking_value in ['enable', 'true']
            } else {
                thinking_mode = bool(data['thinking_mode'])
            }
            logger.info(f"Thinking mode from request: {data['thinking_mode']} -> {thinking_mode}")
        } else if 'enable_thinking' in data {
            // Fallback for legacy parameter
            thinking_mode = bool(data['enable_thinking'])
            logger.info(f"Thinking mode from legacy enable_thinking: {data['enable_thinking']} -> {thinking_mode}")
        } else {
            logger.info("No thinking_mode parameter found in request")
        }
        
        // Add \no_think prefix if thinking mode is disabled
        message_to_send = original_message
        if thinking_mode is False and original_message {
            message_to_send = f"\\no_think {original_message}"
            logger.info("Thinking mode disabled, adding \\no_think prefix to message")
        }
        
        message_data = {
            'model_name': model_name,
            'message': message_to_send,
            'session_id': session.get('session_id'),
            'chat_history': data.get('chat_history', []),
            'generation_config': generation_config,
            'stream': data.get('stream', False)
        }
        
        // Add content filter control
        if data.get('disable_content_filter', False) {
            message_data['disable_content_filter'] = True
        }
        
        if 'content_filter_strictness' in data {
            message_data['content_filter_strictness'] = data['content_filter_strictness']
        }
        
        // Add thinking mode control (keep original thinking_mode for API compatibility)
        if thinking_mode is not None {
            message_data['thinking_mode'] = thinking_mode
        }
        
        async def _send_message():
            async with BeautyAIChatService() as service:
                return await service.send_chat_message(message_data)
        
        logger.info(f"Sending chat message: model={message_data['model_name']}, message_length={len(message_data['message'])}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message())
        loop.close()
        
        logger.info(f"Chat response received: success={result.get('success', False)}")
        
        if result.get('success', False) {
            // Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            performance = generation_stats.get('performance', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': performance.get('tokens_generated', 0),
                'generation_time_ms': performance.get('generation_time_ms', 0),
                'tokens_per_second': performance.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced')
            })
        } else {
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    } catch (Exception e) {
        logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    }

@app.route('/debug')
def debug_page():
    """Debug page for testing thinking mode."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Thinking Mode</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .debug-panel { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .code { background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; margin: 10px 0; white-space: pre-wrap; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
        .iframe-container { margin: 20px 0; }
        iframe { width: 100%; height: 400px; border: 1px solid #ccc; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🔍 Debug Thinking Mode Issue</h1>
    
    <div class="debug-panel">
        <h2>Quick Debug Tests</h2>
        <button onclick="testThinkingMode()">Test Current Thinking Mode</button>
        <button onclick="testAPICall()">Test API Call</button>
        <button onclick="goToMainUI()">Open Main UI</button>
        <div id="results"></div>
    </div>
    
    <div class="debug-panel">
        <h2>Manual Testing Instructions</h2>
        <ol>
            <li><strong>Click "Open Main UI"</strong> to go to the main interface</li>
            <li><strong>Press F12</strong> to open Developer Tools</li>
            <li><strong>Go to Console tab</strong> and copy-paste the debug commands below</li>
            <li><strong>Go to Network tab</strong> to inspect API calls</li>
            <li><strong>Uncheck "Enable Thinking"</strong> in the sidebar</li>
            <li><strong>Send a test message</strong> and check the network request</li>
        </ol>
    </div>
    
    <div class="debug-panel">
        <h2>Console Debug Commands</h2>
        <p>Copy these commands and run them in the browser console on the main UI:</p>
        
        <div class="code">// 1. Check checkbox element
const checkbox = document.getElementById('enable_thinking');
console.log('Checkbox element:', checkbox);
console.log('Checkbox checked:', checkbox?.checked);

// 2. Check parameter controls
const app = window.beautyAIChat || window.app;
console.log('App instance:', app);
console.log('Parameter controls thinking:', app?.parameterControls?.enable_thinking?.checked);

// 3. Monitor checkbox changes
if (checkbox) {
    checkbox.addEventListener('change', (e) => {
        console.log('Checkbox changed to:', e.target.checked);
    });
    console.log('Monitoring checkbox...');
}

// 4. Test payload creation
function testPayload() {
    const checkbox = document.getElementById('enable_thinking');
    const payload = {
        thinking_mode: checkbox?.checked || false
    };
    console.log('Would send payload:', payload);
    return payload;
}
testPayload();</div>
    </div>
    
    <script>
        function testThinkingMode() {
            const results = document.getElementById('results');
            results.innerHTML = '<div class="status info">Testing thinking mode... Check console for details.</div>';
            
            // Test the logic we expect
            const testCases = [
                { checked: true, expected: true },
                { checked: false, expected: false }
            ];
            
            let allPassed = true;
            testCases.forEach(test => {
                const result = test.checked || false;
                const passed = result ===