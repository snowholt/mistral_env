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

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-websocket')
def test_websocket():
    """Serve the WebSocket test page."""
    import os
    test_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'test_websocket_wss.html')
    
    if os.path.exists(test_file_path):
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    else:
        return f"""
        <html>
        <head><title>Test File Not Found</title></head>
        <body>
            <h1>❌ Test File Not Found</h1>
            <p>The WebSocket test file was not found at: <code>{test_file_path}</code></p>
            <p>Expected location: <code>/home/lumi/benchmark_and_test/test_websocket_wss.html</code></p>
            <p><a href="/">← Back to Main UI</a></p>
        </body>
        </html>
        """, 404

@app.route('/api/audio-chat', methods=['POST'])
def audio_chat():
    """Audio chat endpoint."""
    try:
        # Get the audio file
        if 'audio_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400
        
        # Get form data parameters
        form_data = request.form.to_dict()
        
        # Set defaults for required parameters
        model_name = form_data.get('model_name', 'qwen3-unsloth-q4ks')
        whisper_model_name = form_data.get('whisper_model_name', 'whisper-large-v3-turbo-arabic')
        audio_language = form_data.get('audio_language', 'ar')
        
        # Get model info to determine engine type for parameter compatibility
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
        
        # Prepare the audio chat form data
        audio_form_data = {
            'model_name': model_name,
            'whisper_model_name': whisper_model_name,
            'audio_language': audio_language,
            'session_id': session.get('session_id'),
        }
        
        # Add generation parameters
        if 'preset' in form_data:
            audio_form_data['preset'] = form_data['preset']
        
        # Core parameters
        if 'temperature' in form_data:
            audio_form_data['temperature'] = float(form_data['temperature'])
        if 'top_p' in form_data:
            audio_form_data['top_p'] = float(form_data['top_p'])
        if 'repetition_penalty' in form_data:
            audio_form_data['repetition_penalty'] = float(form_data['repetition_penalty'])
        if 'max_new_tokens' in form_data:
            audio_form_data['max_new_tokens'] = int(form_data['max_new_tokens'])
        
        # Engine-specific parameters
        if engine_type == 'llama.cpp':
            if 'top_k' in form_data:
                audio_form_data['top_k'] = int(form_data['top_k'])
            if 'min_p' in form_data:
                audio_form_data['min_p'] = float(form_data['min_p'])
        
        # Content filtering
        if 'disable_content_filter' in form_data:
            audio_form_data['disable_content_filter'] = form_data['disable_content_filter'].lower() == 'true'
        if 'content_filter_strictness' in form_data:
            audio_form_data['content_filter_strictness'] = form_data['content_filter_strictness']
        
        # Thinking mode
        if 'thinking_mode' in form_data:
            # Handle both "true"/"false" and "enable"/"disable" strings
            thinking_value = form_data['thinking_mode'].lower()
            audio_form_data['thinking_mode'] = thinking_value in ['true', 'enable']
        elif 'enable_thinking' in form_data:
            # Fallback for legacy parameter
            audio_form_data['thinking_mode'] = form_data['enable_thinking'].lower() == 'true'
        
        # Chat history
        if 'chat_history' in form_data:
            audio_form_data['chat_history'] = form_data['chat_history']
        
        async def _send_audio_message():
            async with BeautyAIChatService() as service:
                return await service.send_audio_chat_message(audio_file.read(), audio_form_data)
        
        logger.info(f"Sending audio chat message: model={model_name}, whisper_model={whisper_model_name}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_audio_message())
        loop.close()
        
        logger.info(f"Audio chat response received: success={result.get('success', False)}")
        
        if result.get('success', False):
            # Extract response and stats from actual API format
            generation_stats = result.get('generation_stats', {})
            
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'transcription': result.get('transcription', ''),
                'thinking_content': result.get('thinking_content', ''),
                'final_content': result.get('final_content', ''),
                'tokens_generated': result.get('tokens_generated', 0),
                'generation_time_ms': result.get('generation_time_ms', 0),
                'transcription_time_ms': result.get('transcription_time_ms', 0),
                'total_processing_time_ms': result.get('total_processing_time_ms', 0),
                'tokens_per_second': result.get('tokens_per_second', 0),
                'preset_used': result.get('preset_used', ''),
                'content_filter_applied': result.get('content_filter_applied', False),
                'thinking_enabled': result.get('thinking_enabled', False),
                'content_filter_strictness': result.get('content_filter_strictness', 'balanced'),
                'whisper_model_used': result.get('whisper_model_used', whisper_model_name),
                'audio_language_detected': result.get('audio_language_detected', audio_language)
            })
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Audio chat failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'transcription': result.get('transcription', ''),
                'transcription_error': result.get('transcription_error'),
                'chat_error': result.get('chat_error')
            }), 500
            
    except Exception as e:
        logger.error(f"Audio chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)            }), 500

@app.route('/inference/voice-to-voice', methods=['POST'])
def voice_to_voice():
    """Voice-to-Voice conversation endpoint."""
    try:
        # Get the audio file
        if 'audio_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400
        
        # Get form data parameters
        form_data = request.form.to_dict()
        
        # Set defaults for voice-to-voice parameters
        chat_model_name = form_data.get('chat_model_name', 'qwen3-unsloth-q4ks')
        stt_model_name = form_data.get('stt_model_name', 'whisper-large-v3-turbo-arabic')
        tts_model_name = form_data.get('tts_model_name', 'coqui-tts-arabic')
        input_language = form_data.get('input_language', 'auto')
        output_language = form_data.get('output_language', 'auto')
        
        # Prepare the voice-to-voice form data (avoid duplicates)
        v2v_form_data = {
            'chat_model_name': chat_model_name,
            'stt_model_name': stt_model_name,
            'tts_model_name': tts_model_name,
            'input_language': input_language,
            'output_language': output_language,
            'session_id': form_data.get('session_id', session.get('session_id')),
            'audio_output_format': form_data.get('audio_output_format', 'wav'),
        }
        
        # Add voice parameters
        voice_params = ['speaker_voice', 'emotion', 'speech_speed']
        for param in voice_params:
            if param in form_data:
                if param == 'speech_speed':
                    v2v_form_data[param] = float(form_data[param])
                else:
                    v2v_form_data[param] = form_data[param]
        
        # Add generation parameters (preset takes priority)
        preset = form_data.get('preset')
        if preset:
            v2v_form_data['preset'] = preset
        
        # Add individual generation parameters only if no preset or custom preset
        if not preset or preset == 'custom':
            gen_params = ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens', 'min_p']
            for param in gen_params:
                if param in form_data and form_data[param]:
                    try:
                        if param in ['top_k', 'max_new_tokens']:
                            v2v_form_data[param] = int(form_data[param])
                        else:
                            v2v_form_data[param] = float(form_data[param])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {param}: {form_data[param]}")
        
        # Content filtering (ensure no duplicates)
        if 'enable_content_filter' in form_data:
            # Convert to disable_content_filter for backend compatibility
            v2v_form_data['disable_content_filter'] = form_data['enable_content_filter'].lower() == 'false'
        elif 'disable_content_filter' in form_data:
            v2v_form_data['disable_content_filter'] = form_data['disable_content_filter'].lower() == 'true'
        
        if 'content_filter_strictness' in form_data and form_data['content_filter_strictness']:
            v2v_form_data['content_filter_strictness'] = form_data['content_filter_strictness']
        
        # Thinking mode
        if 'thinking_mode' in form_data:
            v2v_form_data['thinking_mode'] = form_data['thinking_mode'].lower() == 'true'
        
        # Chat history
        if 'chat_history' in form_data and form_data['chat_history']:
            v2v_form_data['chat_history'] = form_data['chat_history']
        
        async def _send_voice_to_voice_message():
            async with BeautyAIChatService() as service:
                return await service.send_voice_to_voice_message(audio_file.read(), v2v_form_data)
        
        logger.info(f"Sending voice-to-voice message: chat_model={chat_model_name}, stt_model={stt_model_name}, tts_model={tts_model_name}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_voice_to_voice_message())
        loop.close()
        
        logger.info(f"Voice-to-voice response received: success={result.get('success', False)}")
        
        if result.get('success', False):
            # Return the audio response directly
            response_data = {
                'success': True,
                'transcription': result.get('transcription', ''),
                'response_text': result.get('response_text', ''),
                'audio_output_format': result.get('audio_output_format', 'wav'),
                'audio_size_bytes': result.get('audio_size_bytes', 0),
                'session_id': result.get('session_id', ''),
                'total_processing_time_ms': result.get('total_processing_time_ms', 0),
                'transcription_time_ms': result.get('transcription_time_ms', 0),
                'generation_time_ms': result.get('generation_time_ms', 0),
                'audio_generation_time_ms': result.get('audio_generation_time_ms', 0),
                'models_used': result.get('models_used', {}),
                'content_filter_applied': result.get('content_filter_applied', False),
                'preset_used': result.get('preset_used', ''),
                'speaker_voice_used': result.get('speaker_voice_used', ''),
                'emotion_used': result.get('emotion_used', ''),
                'speech_speed_used': result.get('speech_speed_used', 1.0)
            }
            
            # If we have audio data, we need to handle it differently
            if 'audio_data' in result:
                # Return audio as base64 for easier handling in JavaScript
                import base64
                audio_data = result['audio_data']
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                response_data['audio_data'] = audio_base64
            elif result.get('audio_size_bytes', 0) > 0:
                # QUICK FIX: Audio is missing from response but was generated
                # Log this so we know the issue exists
                logger.warning(f"Audio was generated ({result.get('audio_size_bytes')} bytes) but not included in response for session {result.get('session_id')}")
                logger.warning("This indicates the BeautyAI API needs to be updated to include 'audio_data' in the JSON response")
                
                # Add a note to the response so the frontend knows what happened
                response_data['audio_missing'] = True
                response_data['audio_missing_reason'] = "Audio generated but not included in API response"
                
            return jsonify(response_data)
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Voice-to-voice failed: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'transcription_error': result.get('transcription_error'),
                'generation_error': result.get('generation_error'),
                'tts_error': result.get('tts_error'),
                'errors': result.get('errors', [])
            }), 500
            
    except Exception as e:
        logger.error(f"Voice-to-voice error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/audio-download/<session_id>')
def audio_download(session_id):
    """Try to download audio for a specific session ID."""
    try:
        # Try various possible audio endpoints on the BeautyAI API
        audio_endpoints = [
            f"{BEAUTYAI_API_URL}/audio/{session_id}",
            f"{BEAUTYAI_API_URL}/audio/{session_id}.wav",
            f"{BEAUTYAI_API_URL}/inference/audio/{session_id}",
            f"{BEAUTYAI_API_URL}/download/audio/{session_id}",
            f"{BEAUTYAI_API_URL}/tts/audio/{session_id}",
        ]
        
        import requests
        for endpoint in audio_endpoints:
            try:
                logger.info(f"Trying audio endpoint: {endpoint}")
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200 and response.headers.get('content-type', '').startswith('audio/'):
                    logger.info(f"Successfully fetched audio from {endpoint}")
                    return Response(
                        response.content,
                        mimetype=response.headers.get('content-type', 'audio/wav'),
                        headers={
                            'Content-Length': len(response.content),
                            'Cache-Control': 'no-cache'
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to fetch from {endpoint}: {e}")
                continue
        
        # If no audio endpoint works, return 404
        return jsonify({
            'success': False,
            'error': f'No audio found for session {session_id}'
        }), 404
        
    except Exception as e:
        logger.error(f"Audio download error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ---------------------------------------------------------------------------
# Debug: PCM Upload → WebSocket streaming test page (no microphone)
# ---------------------------------------------------------------------------
@app.route('/debug/pcm-upload')
def debug_pcm_upload_page():
    """Serve a standalone HTML page for uploading a raw 16 kHz mono s16le
    PCM file (.pcm) and streaming it over the BeautyAI streaming voice
    WebSocket endpoint. No microphone required. Intentionally self-contained
    for rapid debugging of partial/final transcripts and TTS events.
    """
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>BeautyAI PCM → WebSocket Debug</title>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
        body { font-family: system-ui, Arial, sans-serif; margin: 0; padding: 0; background:#f5f7fa; color:#222; }
        header { background:#3f51b5; color:#fff; padding:16px 24px; }
        h1 { margin:0; font-size:20px; }
        main { max-width: 1100px; margin: 0 auto; padding: 24px; }
        section { background:#fff; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.07); padding:20px; margin-bottom:24px; }
        label { display:block; margin: 8px 0 4px; font-weight:600; }
        input[type=text], select, input[type=number] { width:100%; padding:8px 10px; border:1px solid #ccc; border-radius:6px; font-size:14px; }
        input[type=file] { margin-top:4px; }
    button { cursor:pointer; border:none; background:#3f51b5; color:#fff; padding:10px 18px; border-radius:6px; font-size:14px; font-weight:600; letter-spacing:.3px; display:inline-flex; align-items:center; gap:6px; }
        button:disabled { background:#9fa8da; cursor:not-allowed; }
        button.secondary { background:#607d8b; }
    button.outline { background:#fff; color:#3f51b5; border:1px solid #3f51b5; }
    button.outline:hover { background:#3f51b5; color:#fff; }
        .row { display:flex; flex-wrap:wrap; gap:18px; }
        .col { flex:1 1 260px; min-width:260px; }
        .log { background:#0d1117; color:#e6edf3; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding:12px; height:340px; overflow:auto; border-radius:6px; font-size:13px; line-height:1.4; }
        .status-bar { display:flex; flex-wrap:wrap; gap:10px; margin:12px 0; font-size:13px; }
        .status-item { background:#e8eef5; padding:6px 10px; border-radius:5px; }
        .events-table { width:100%; border-collapse:collapse; font-size:13px; }
        .events-table th, .events-table td { border:1px solid #ddd; padding:4px 6px; text-align:left; }
        .events-table th { background:#f1f5f9; }
        .pill { display:inline-block; padding:2px 6px; border-radius:12px; font-size:11px; font-weight:600; }
        .pill.partial { background:#fff8e1; color:#9c6500; }
        .pill.final { background:#e0f7fa; color:#006064; }
        .pill.tts { background:#ede7f6; color:#4527a0; }
        audio { width:100%; margin-top:8px; }
        .progress { height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden; margin-top:6px; }
        .progress span { display:block; height:6px; background:#3f51b5; width:0%; transition: width .2s; }
    </style>
</head>
<body>
    <header><h1>BeautyAI PCM → WebSocket Streaming Debug</h1></header>
    <main>
        <section>
            <h2>1. Select PCM File</h2>
            <p>Upload a <strong>raw 16 kHz mono signed 16-bit little-endian PCM</strong> file (.pcm). Frames are sent over the WebSocket to debug partial/final transcripts & TTS without microphone.</p>
            <div class="row">
                <div class="col"><label for="pcmFile">PCM File (.pcm)</label><input type="file" id="pcmFile" accept=".pcm" /></div>
                <div class="col"><label for="language">Language</label><select id="language"><option value="en">en</option><option value="ar">ar</option></select></div>
                <div class="col"><label for="frameMs">Frame (ms)</label><input type="number" id="frameMs" value="20" min="10" max="200" step="5" /></div>
                <div class="col"><label for="endpointUrl">WebSocket URL</label><input type="text" id="endpointUrl" value="wss://api.gmai.sa/api/v1/ws/streaming-voice" /></div>
            </div>
            <div class="row" style="margin-top:12px;">
                <div class="col"><label for="tailSilenceMs">Tail Silence (ms)</label><input type="number" id="tailSilenceMs" value="1200" min="0" step="100" /></div>
                <div class="col"><label for="autoCloseSec">Auto-Close After (s)</label><input type="number" id="autoCloseSec" value="6" min="0" step="0.5" /></div>
                                <div class="col"><label>&nbsp;</label>
                                    <div class="flex-col">
                                        <button id="connectBtn">Connect & Start</button>
                                        <div style="margin-top:8px; font-size:11px; color:#444; line-height:1.3;">Tip: Set Auto-Close=0 to keep socket open until you manually abort or server closes after TTS.</div>
                                    </div>
                                </div>
                                <div class="col"><label>&nbsp;</label>
                                    <div class="flex-col" style="gap:6px;display:flex;">
                                        <button id="abortBtn" class="secondary" disabled>Abort / Close</button>
                                        <button id="exportBtn" class="outline" disabled>Export Report</button>
                                    </div>
                                </div>
            </div>
            <div class="progress"><span id="progressBar"></span></div>
            <div class="status-bar" id="statusBar"></div>
                        <details style="margin-top:10px;">
                            <summary><strong>Auto-Close Behavior</strong></summary>
                            <p style="font-size:13px; line-height:1.5;">If you choose a non-zero Auto-Close value the page will schedule a closure AFTER audio streaming finishes. If a <code>tts_start</code> event arrives the timer is extended to allow TTS generation. If no <code>tts_audio</code> or <code>tts_complete</code> arrives before the extended timeout you'll see a warning in the report. Code 1005 (no close frame) usually means the browser or proxy cut the connection before a normal close handshake.</p>
                        </details>
        </section>
        <section>
            <h2>2. Events</h2>
            <table class="events-table" id="eventsTable"><thead><tr><th>#</th><th>Type</th><th>Idx</th><th>Text / Info</th><th>Latency (ms)</th></tr></thead><tbody></tbody></table>
        </section>
        <section>
            <h2>3. Raw JSON Log</h2>
            <div class="log" id="log"></div>
        </section>
        <section>
            <h2>4. TTS Audio</h2>
            <div id="ttsContainer"><em>No TTS yet.</em></div>
        </section>
    </main>
    <script>
        const logEl=document.getElementById('log');
        const eventsBody=document.querySelector('#eventsTable tbody');
        const statusBar=document.getElementById('statusBar');
        const progressBar=document.getElementById('progressBar');
        const ttsContainer=document.getElementById('ttsContainer');
    const connectBtn=document.getElementById('connectBtn');
    const abortBtn=document.getElementById('abortBtn');
    const exportBtn=document.getElementById('exportBtn');
    let ws=null,startTs=null,partialCount=0,finalCount=0,ttsCount=0;
    let audioStreamFinished=false;
    let ttsStarted=false;
    let ttsAudioReceived=false;
    let autoCloseTimer=null;
    let eventsCollected=[]; // raw event objects (including meta)
    let lastEventTs=null;
        function addStatus(k,v){const d=document.createElement('div');d.className='status-item';d.textContent=`${k}: ${v}`;statusBar.appendChild(d);}function clearStatus(){statusBar.innerHTML='';}
    function logJson(o){eventsCollected.push({ts:Date.now(), ...o});const line=document.createElement('div');line.textContent=JSON.stringify(o);logEl.appendChild(line);logEl.scrollTop=logEl.scrollHeight;lastEventTs=Date.now();}
        function addEventRow(e){const tr=document.createElement('tr');const lat=startTs?Date.now()-startTs:0;const idx=e.cursor??e.utterance_index??'';let pc='';if(e.type==='partial_transcript')pc='partial';else if(e.type==='final_transcript')pc='final';else if(e.type && e.type.startsWith('tts_'))pc='tts';tr.innerHTML=`<td>${eventsBody.children.length+1}</td><td><span class="pill ${pc}">${e.type}</span></td><td>${idx}</td><td>${(e.text||e.message||'').slice(0,140)}</td><td>${lat}</td>`;eventsBody.appendChild(tr);} 
        function decodeB64Wav(b){const bin=atob(b);const len=bin.length;const bytes=new Uint8Array(len);for(let i=0;i<len;i++)bytes[i]=bin.charCodeAt(i);return new Blob([bytes],{type:'audio/wav'});} 
                function scheduleAutoClose(label, seconds){ if(seconds<=0) return; if(autoCloseTimer){ clearTimeout(autoCloseTimer); autoCloseTimer=null; }
                    autoCloseTimer=setTimeout(()=>{ if(ws&&ws.readyState===WebSocket.OPEN){ logJson({event:'auto_close', reason:label}); ws.close(); } }, seconds*1000); }
                function streamPcm(pcmBytes,frameSamples,frameMs,tailSilenceMs,autoCloseSec){const total=pcmBytes.byteLength/2;let off=0;function step(){if(!ws||ws.readyState!==WebSocket.OPEN)return;if(off<total){const remain=total-off;const thisS=Math.min(remain,frameSamples);const slice=pcmBytes.slice(off*2,(off+thisS)*2);ws.send(slice);off+=thisS;progressBar.style.width=`${Math.min(100,(off/total)*100)}%`;setTimeout(step,frameMs);}else{audioStreamFinished=true; if(tailSilenceMs>0){const silenceFrames=Math.max(1,Math.round(tailSilenceMs/frameMs));const silence=new Uint8Array(frameSamples*2);for(let i=0;i<silenceFrames;i++)ws.send(silence);}progressBar.style.width='100%'; if(autoCloseSec>0) scheduleAutoClose('post_audio', autoCloseSec); } }step();}
                connectBtn.addEventListener('click',()=>{const f=document.getElementById('pcmFile');if(!f.files.length){alert('Select a .pcm file');return;}eventsCollected=[];ttsStarted=false;ttsAudioReceived=false;audioStreamFinished=false;partialCount=0;finalCount=0;ttsCount=0;exportBtn.disabled=true;const lang=document.getElementById('language').value.trim();const frameMs=parseInt(document.getElementById('frameMs').value,10)||20;const endpoint=document.getElementById('endpointUrl').value.trim();const tailSilenceMs=parseInt(document.getElementById('tailSilenceMs').value,10)||0;const autoCloseSec=parseFloat(document.getElementById('autoCloseSec').value)||0;const file=f.files[0];const url=`${endpoint}?language=${encodeURIComponent(lang)}`;ws=new WebSocket(url);ws.binaryType='arraybuffer';startTs=Date.now();clearStatus();addStatus('State','Connecting');connectBtn.disabled=true;abortBtn.disabled=false;logJson({event:'connecting',url,frame_ms:frameMs,tail_silence_ms:tailSilenceMs,auto_close_sec:autoCloseSec});ws.onopen=()=>{clearStatus();addStatus('State','Open');logJson({event:'open'});file.arrayBuffer().then(buf=>{const frameSamples=Math.round(16000*(frameMs/1000));streamPcm(new Uint8Array(buf),frameSamples,frameMs,tailSilenceMs,autoCloseSec);});};ws.onmessage=ev=>{let d;try{d=JSON.parse(ev.data);}catch{return;}logJson(d);addEventRow(d);if(d.type==='partial_transcript')partialCount++;if(d.type==='final_transcript')finalCount++;if(d.type==='tts_start'){ttsStarted=true; if(autoCloseSec>0){ // extend timer if needed
                            scheduleAutoClose('tts_wait_extension', Math.max(8, autoCloseSec));
                        }}
                        if(d.type==='tts_audio'&&d.audio){ttsCount++;ttsAudioReceived=true;const blob=decodeB64Wav(d.audio);const url=URL.createObjectURL(blob);const audio=document.createElement('audio');audio.controls=true;audio.src=url;const meta=document.createElement('div');meta.textContent=`TTS #${ttsCount} (${blob.size} bytes, chars=${d.chars||''})`;ttsContainer.prepend(meta);ttsContainer.prepend(audio);}
                        if(d.type==='tts_complete'){ttsAudioReceived=true; if(autoCloseSec>0){ scheduleAutoClose('tts_complete_grace', 2); }}
                        clearStatus();addStatus('State','Open');addStatus('Partials',partialCount);addStatus('Finals',finalCount);addStatus('TTS',ttsCount);};ws.onerror=e=>{logJson({event:'error',message:'WebSocket error',detail:e.message||String(e)});addStatus('Error','Yes');};ws.onclose=e=>{logJson({event:'close',code:e.code,reason:e.reason});clearStatus();addStatus('State','Closed');addStatus('Code',e.code);if(partialCount||finalCount)addStatus('Totals',`P=${partialCount},F=${finalCount},TTS=${ttsCount}`);if(ttsStarted&&!ttsAudioReceived){addStatus('Warning','TTS started but no audio (increase Auto-Close or check server logs)');logJson({event:'diagnostic',warning:'tts_started_no_audio',suggestion:'Set Auto-Close=0 or larger, inspect server TTS logs'});}connectBtn.disabled=false;abortBtn.disabled=true;exportBtn.disabled=false;};});
        abortBtn.addEventListener('click',()=>{if(ws&&ws.readyState===WebSocket.OPEN)ws.close();});
                exportBtn.addEventListener('click',()=>{ if(!eventsCollected.length){return;} const meta={ generated_at:new Date().toISOString(), total_events:eventsCollected.length, partials:partialCount, finals:finalCount, tts_audio:ttsCount, tts_started:ttsStarted, tts_audio_received:ttsAudioReceived, auto_close_timer_present:!!autoCloseTimer, duration_ms: lastEventTs && startTs ? (lastEventTs-startTs): null }; const report={ meta, events:eventsCollected }; const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; const fname=`beautyai_ws_debug_${new Date().toISOString().replace(/[:.]/g,'-')}.json`; a.download=fname; document.body.appendChild(a); a.click(); setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 1000); });
    </script>
</body></html>'''
    return Response(html, mimetype='text/html')

# ---------------------------------------------------------------------------
# Hybrid Microphone Streaming Debug Page (Client PCM + Fallback)
# ---------------------------------------------------------------------------
@app.route('/debug/mic-hybrid')
def debug_mic_hybrid():
    """Serve a page that attempts AudioWorklet capture -> 16k PCM frames.
    Falls back to MediaRecorder (webm) if worklet unsupported.
    """
    html = r'''<!DOCTYPE html><html><head><meta charset="utf-8" />
    <title>BeautyAI Hybrid Mic Streaming</title>
    <style>body{font-family:system-ui,Arial;margin:0;padding:18px;background:#f5f7fa;color:#222}h1{margin-top:0}.log{background:#0d1117;color:#e6edf3;font:13px ui-monospace;white-space:pre-wrap;padding:10px;border-radius:6px;height:240px;overflow:auto}button{background:#3f51b5;color:#fff;border:none;padding:10px 18px;border-radius:6px;margin:4px;cursor:pointer;font-weight:600}button.secondary{background:#607d8b}#status span{display:inline-block;background:#e3e8ef;padding:4px 8px;border-radius:4px;margin:2px;font-size:12px}fieldset{border:1px solid #cfd8dc;border-radius:8px;margin-bottom:18px;padding:12px}legend{padding:0 6px;font-weight:600}label{display:block;font-size:13px;margin-top:6px}input,select{padding:6px 8px;border:1px solid #b0bec5;border-radius:4px;width:100%;font-size:14px;box-sizing:border-box}.row{display:flex;gap:14px;flex-wrap:wrap}.col{flex:1 1 180px;min-width:180px}audio{margin-top:8px;width:100%}.pill{display:inline-block;padding:2px 6px;font-size:11px;border-radius:10px;background:#eceff1;margin-right:4px}</style>
    </head><body><h1>Hybrid Microphone Streaming Debug</h1>
    <p>This page tries to stream raw 16 kHz PCM frames via an AudioWorklet. If unavailable, it falls back to MediaRecorder (WebM/Opus). The backend will auto-detect and decode WebM when allowed.</p>
    <fieldset><legend>Connection</legend>
      <div class="row">
        <div class="col"><label>WebSocket URL<input id="wsUrl" value="wss://api.gmai.sa/api/v1/ws/streaming-voice"></label></div>
        <div class="col"><label>Language<select id="lang"><option value="en">en</option><option value="ar">ar</option></select></label></div>
        <div class="col"><label>Frame (ms)<input id="frameMs" type="number" value="20" min="10" max="120" step="5"></label></div>
        <div class="col"><label>Max Duration (s)<input id="maxSec" type="number" value="12" min="2" max="120"></label></div>
      </div>
      <div><button id="startBtn">Start</button><button id="stopBtn" class="secondary" disabled>Stop</button><button id="clearBtn" class="secondary">Clear Log</button></div>
      <div id="status"></div>
    </fieldset>
    <fieldset><legend>Events</legend><div class="log" id="log"></div></fieldset>
    <fieldset><legend>TTS Audio</legend><div id="tts"></div></fieldset>
    <script>
    const logEl=document.getElementById('log');
    const statusEl=document.getElementById('status');
    const ttsEl=document.getElementById('tts');
    const startBtn=document.getElementById('startBtn');
    const stopBtn=document.getElementById('stopBtn');
    const clearBtn=document.getElementById('clearBtn');
    let ws=null;let workletSupported=false;let mediaStream=null;let mediaRecorder=null;let audioCtx=null;let pcmNodePort=null;let closed=false;let chunks=[];let frameTimer=null;let startTs=null;let ttsCount=0;let rawMode='';
    function log(o){if(typeof o!=='string')o=JSON.stringify(o);logEl.textContent+=o+'\n';logEl.scrollTop=logEl.scrollHeight;}
    function putStatus(k,v){const span=document.createElement('span');span.textContent=`${k}:${v}`;statusEl.appendChild(span);} function clearStatus(){statusEl.innerHTML='';}
    async function start(){
        clearStatus();closed=false;chunks=[];ttsEl.innerHTML='';
        const url=document.getElementById('wsUrl').value.trim()+`?language=${encodeURIComponent(document.getElementById('lang').value)}`;
        const frameMs=parseInt(document.getElementById('frameMs').value,10)||20;
        const maxSec=parseFloat(document.getElementById('maxSec').value)||12;
        putStatus('connecting', '...');
        ws=new WebSocket(url); ws.binaryType='arraybuffer';
        ws.onopen=()=>{clearStatus();putStatus('open','1');log({event:'open'});initCapture(frameMs,maxSec);};
        ws.onmessage=(ev)=>{let d;try{d=JSON.parse(ev.data);}catch{return;}log(d);if(d.type==='partial_transcript'){putStatus('partial',Date.now()-startTs);} if(d.type==='final_transcript'){putStatus('final',Date.now()-startTs);} if(d.type==='tts_audio'&&d.audio){ttsCount++; const b64=d.audio; const bin=atob(b64); const bytes=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i); const blob=new Blob([bytes],{type:'audio/wav'}); const a=document.createElement('audio'); a.controls=true; a.src=URL.createObjectURL(blob); const meta=document.createElement('div'); meta.textContent=`TTS #${ttsCount} (${blob.size} bytes)`; ttsEl.prepend(meta); ttsEl.prepend(a);} };
        ws.onerror=e=>{log({event:'error',detail:e.message||String(e)});};
        ws.onclose=e=>{log({event:'close',code:e.code,reason:e.reason}); cleanup();};
        startBtn.disabled=true; stopBtn.disabled=false;
        startTs=Date.now();
    }
    async function initCapture(frameMs,maxSec){
        try{mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});}catch(e){log({event:'mic_error',message:e.message}); return;}
        // Try worklet
        if(window.AudioWorkletNode){
            try{
                audioCtx=new AudioContext();
                await audioCtx.audioWorklet.addModule('/worklet/pcm_downsampler.js');
                const src=audioCtx.createMediaStreamSource(mediaStream);
                const node=new AudioWorkletNode(audioCtx,'pcm-downsampler',{processorOptions:{targetSampleRate:16000, frameSamples:Math.round(16000*(frameMs/1000))}});
                pcmNodePort=node.port; workletSupported=true; rawMode='worklet';
                node.port.onmessage=(e)=>{ if(ws && ws.readyState===WebSocket.OPEN){ ws.send(e.data); } };
                src.connect(node); // no destination to stay silent
                log({event:'capture_mode',mode:'worklet_pcm'});
                putStatus('mode','worklet');
                setTimeout(()=>{ if(!closed) stop(); }, maxSec*1000);
                return;
            }catch(e){ log({event:'worklet_fail',message:e.message}); }
        }
        // Fallback: MediaRecorder (webm opus)
        try {
            mediaRecorder=new MediaRecorder(mediaStream,{mimeType:'audio/webm'});
            rawMode='mediarecorder'; putStatus('mode','webm');
            mediaRecorder.ondataavailable=ev=>{ if(ev.data && ev.data.size>0 && ws && ws.readyState===WebSocket.OPEN){ ev.data.arrayBuffer().then(buf=>{ ws.send(buf); }); } };
            mediaRecorder.start(frameMs); // timeslice in ms; may clamp
            log({event:'capture_mode',mode:'mediarecorder_webm'});
            setTimeout(()=>{ if(!closed) stop(); }, maxSec*1000);
        } catch(e){ log({event:'fallback_fail',message:e.message}); }
    }
    function stop(){closed=true; if(mediaRecorder && mediaRecorder.state!=='inactive'){mediaRecorder.stop();} if(audioCtx){audioCtx.close();} if(ws && ws.readyState===WebSocket.OPEN){ws.close();} cleanup();}
    function cleanup(){stopBtn.disabled=true; startBtn.disabled=false;}
    startBtn.onclick=start; stopBtn.onclick=stop; clearBtn.onclick=()=>{logEl.textContent='';};
    </script></body></html>'''
    return Response(html, mimetype='text/html')

# ---------------------------------------------------------------------------
# Streaming Voice Dedicated Debug Page (English only, pure client PCM)
# ---------------------------------------------------------------------------
@app.route('/debug/streaming-live')
def debug_streaming_live():
        """A focused page for debugging streaming voice with ONLY:
        - English language (no auto / overlay / legacy fallback)
        - Client-side AudioWorklet capture -> 16k PCM Int16 frames
        - Real-time mic level bar
        - Ingest mode / connection status / sample rate display
        - Event log + partial/final transcript panels
        - TTS audio auto-play (handled by StreamingVoiceClient)
        This bypasses MediaRecorder entirely and helps isolate overlay issues.
        """
        html = r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />
<title>BeautyAI Streaming Voice Live Debug (EN)</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
body{font-family:system-ui,Arial,sans-serif;margin:0;background:#10151c;color:#f1f5f9;}
header{background:#1e293b;padding:14px 22px;display:flex;align-items:center;gap:16px;}
h1{margin:0;font-size:18px;font-weight:600;color:#fff;letter-spacing:.5px;}
main{max-width:1200px;margin:0 auto;padding:20px 24px;display:grid;grid-template-columns:320px 1fr;gap:20px;}
section{background:#16202b;border:1px solid #1f2935;border-radius:10px;padding:16px;}
section h2{margin:0 0 10px;font-size:14px;letter-spacing:.6px;text-transform:uppercase;color:#8fb3ff;font-weight:600;}
button{cursor:pointer;border:none;border-radius:6px;font-weight:600;font-size:13px;letter-spacing:.4px;padding:8px 14px;background:#2563eb;color:#fff;display:inline-flex;align-items:center;gap:6px;}
button.secondary{background:#334155;}
button.danger{background:#b91c1c;}
button:disabled{opacity:.55;cursor:not-allowed;}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;}
.status-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px;margin-top:6px;}
.status-box{background:#1e2a36;padding:6px 8px;border-radius:6px;font-size:11px;line-height:1.3;}
.status-box span{display:block;font-size:10px;opacity:.65;letter-spacing:.5px;text-transform:uppercase;}
#micBarWrap{height:28px;background:#1e2a36;border-radius:6px;overflow:hidden;position:relative;margin:8px 0 4px;}
#micBar{position:absolute;left:0;top:0;bottom:0;width:0%;background:linear-gradient(90deg,#22c55e,#f59e0b,#dc2626);transition:width .12s;}
#micLevelText{font-size:11px;opacity:.75;letter-spacing:.4px;}
.live-partial{font-style:italic;background:#1e2a36;padding:10px 12px;border-radius:6px;margin:0 0 10px;font-size:13px;color:#e2e8f0;}
.transcripts-log{max-height:240px;overflow:auto;font-size:13px;line-height:1.45;display:flex;flex-direction:column;gap:8px;}
.utt-user{background:#1e2a36;padding:8px 10px;border-radius:6px;align-self:flex-end;}
.utt-assistant{background:#233242;padding:8px 10px;border-radius:6px;}
table{width:100%;border-collapse:collapse;font-size:11px;}
th,td{border:1px solid #223040;padding:4px 6px;text-align:left;}
th{background:#1b2733;font-weight:600;letter-spacing:.4px;}
#eventsTableWrapper{max-height:280px;overflow:auto;}
.pill{display:inline-block;padding:2px 6px;border-radius:10px;font-weight:600;font-size:10px;letter-spacing:.5px;}
.pill.partial{background:#facc1533;color:#fbbf24;}
.pill.final{background:#10b98133;color:#34d399;}
.pill.tts{background:#6366f133;color:#818cf8;}
.pill.sys{background:#475569;color:#cbd5e1;}
.log-line{white-space:nowrap;}
.flex{display:flex;gap:8px;align-items:center;}
.mono{font-family:ui-monospace,monospace;font-size:12px;}
.badge{background:#334155;color:#cbd5e1;padding:2px 6px;border-radius:4px;font-size:11px;letter-spacing:.4px;}
a.export{font-size:11px;color:#60a5fa;text-decoration:none;margin-left:8px;}
@media (max-width:900px){main{grid-template-columns:1fr;}section#left{order:2;}section#right{order:1;}}
</style>
</head><body>
<header><h1>Streaming Voice Live Debug (EN)</h1><span class="badge" id="connState">idle</span><span class="badge" id="ingestMode">mode: -</span><span class="badge" id="sampleRate">sr: -</span></header>
<main>
    <section id="left">
        <h2>Microphone & Status</h2>
        <div class="controls">
            <button id="btnStart">Start</button>
            <button id="btnSuspend" class="secondary" disabled>Suspend</button>
            <button id="btnResume" class="secondary" disabled>Resume</button>
            <button id="btnDisconnect" class="danger" disabled>Disconnect</button>
            <button id="btnClear" class="secondary">Clear</button>
        </div>
        <div id="micBarWrap"><div id="micBar"></div></div>
        <div id="micLevelText">Mic: --</div>
        <div class="status-grid" id="statusGrid"></div>
        <h2 style="margin-top:20px;">Live Partial</h2>
        <div id="livePartial" class="live-partial" style="display:none;"></div>
        <h2>Final Transcripts</h2>
        <div class="transcripts-log" id="transcriptsLog"></div>
        <h2 style="margin-top:20px;">TTS Audio</h2>
        <div id="ttsAudio"><em>No TTS yet.</em></div>
    </section>
    <section id="right">
        <h2>Events</h2>
        <div id="eventsTableWrapper">
            <table id="eventsTable"><thead><tr><th>#</th><th>Type</th><th>Idx</th><th>Info</th><th>Latency</th></tr></thead><tbody></tbody></table>
        </div>
        <h2 style="margin-top:18px;">Raw Log <a id="exportLink" class="export" href="#" style="display:none;">Export</a></h2>
        <div style="background:#0f1720;border:1px solid #1e2a33;border-radius:6px;padding:8px;height:180px;overflow:auto;font:11px/1.4 ui-monospace,monospace;" id="rawLog"></div>
    </section>
</main>
<script>
// Feature flag just for consistency
window.BEAUTYAI_STREAMING_VOICE = true;
(function(){ if(!window.BEAUTYAI_API_HOST){ const hn=location.hostname; if(/\.gmai\.sa$/i.test(hn) && !/^api\./i.test(hn)){ window.BEAUTYAI_API_HOST='api.gmai.sa'; } } })();
</script>
<script src="/static/js/streamingVoiceClient.js"></script>
<script>
const btnStart=document.getElementById('btnStart');
const btnSuspend=document.getElementById('btnSuspend');
const btnResume=document.getElementById('btnResume');
const btnDisconnect=document.getElementById('btnDisconnect');
const btnClear=document.getElementById('btnClear');
const micBar=document.getElementById('micBar');
const micLevelText=document.getElementById('micLevelText');
const connState=document.getElementById('connState');
const ingestModeBadge=document.getElementById('ingestMode');
const sampleRateBadge=document.getElementById('sampleRate');
const statusGrid=document.getElementById('statusGrid');
const livePartialDiv=document.getElementById('livePartial');
const transcriptsLog=document.getElementById('transcriptsLog');
const ttsAudio=document.getElementById('ttsAudio');
const eventsBody=document.querySelector('#eventsTable tbody');
const rawLog=document.getElementById('rawLog');
const exportLink=document.getElementById('exportLink');
let client=null; let startTs=null; let eventCounter=0; let finalCount=0; let partialCount=0; let ttsCount=0; let rawEvents=[];

function addStatus(k,v){ const box=document.createElement('div'); box.className='status-box'; box.innerHTML='<span>'+k+'</span>'+v; statusGrid.appendChild(box); }
function refreshStatus(){ statusGrid.innerHTML=''; addStatus('Partials',partialCount); addStatus('Finals',finalCount); addStatus('TTS',ttsCount); }
function logRaw(o){ rawEvents.push({ts:Date.now(), ...o}); const line=document.createElement('div'); line.className='log-line'; line.textContent=JSON.stringify(o); rawLog.appendChild(line); rawLog.scrollTop=rawLog.scrollHeight; exportLink.style.display='inline'; }
function addEventRow(evt){ eventCounter++; const tr=document.createElement('tr'); const lat=startTs? (Date.now()-startTs):0; const pillClass= evt.type==='partial'? 'partial': evt.type==='final'? 'final': (evt.type && evt.type.startsWith('tts_'))? 'tts':'sys'; const idx = evt.utterance_index ?? evt.cursor ?? ''; const info=(evt.text||evt.message||'').slice(0,120); tr.innerHTML=`<td>${eventCounter}</td><td><span class="pill ${pillClass}">${evt.type}</span></td><td>${idx}</td><td>${info}</td><td>${lat}</td>`; eventsBody.appendChild(tr); }
function showPartial(text){ if(!text){ livePartialDiv.style.display='none'; livePartialDiv.textContent=''; return; } livePartialDiv.style.display='block'; livePartialDiv.textContent=text; }
function addFinal(text){ const d=document.createElement('div'); d.className='utt-user'; d.textContent=text||'(empty)'; transcriptsLog.appendChild(d); transcriptsLog.scrollTop=transcriptsLog.scrollHeight; }
function addAssistant(text){ const d=document.createElement('div'); d.className='utt-assistant'; d.textContent=text||'(no text)'; transcriptsLog.appendChild(d); transcriptsLog.scrollTop=transcriptsLog.scrollHeight; }

function handleEvent(ev){
    switch(ev.type){
        case 'ws_open': connState.textContent='opening'; logRaw({evt:'ws_open'}); break;
        case 'ready': connState.textContent='ready'; logRaw({evt:'ready'}); addEventRow({type:'ready'}); break;
        case 'mic_level': const pct=Math.round(Math.min(1,ev.level*4)*100); micBar.style.width=pct+'%'; micLevelText.textContent='Mic RMS ~ '+pct+'%'; break;
        case 'partial': partialCount++; showPartial(ev.text); addEventRow({type:'partial', text:ev.text}); refreshStatus(); break;
        case 'final': finalCount++; showPartial(''); addFinal(ev.text); addEventRow({type:'final', text:ev.text, utterance_index:ev.utterance_index}); refreshStatus(); break;
        case 'ingest_mode': ingestModeBadge.textContent='mode: '+ev.mode; logRaw({evt:'ingest_mode', mode:ev.mode}); addEventRow({type:'ingest_mode', text:ev.mode}); break;
        case 'ingest_summary': logRaw({evt:'ingest_summary', summary:ev.summary}); addEventRow({type:'ingest_summary', text: (ev.summary&&ev.summary.mode)||''}); break;
        case 'tts_audio': ttsCount++; refreshStatus(); addEventRow({type:'tts_audio', text:'chars='+ev.chars}); break;
        case 'tts_complete': addEventRow({type:'tts_complete'}); break;
        case 'error': connState.textContent='error'; logRaw({evt:'error', message:ev.message, stage:ev.stage}); addEventRow({type:'error', text:ev.message||ev.stage||''}); break;
        case 'ws_close': connState.textContent='closed'; logRaw({evt:'ws_close', code:ev.code, reason:ev.reason}); addEventRow({type:'ws_close', text:ev.reason||''}); btnSuspend.disabled=true; btnResume.disabled=true; btnDisconnect.disabled=true; break;
        default: break;
    }
}

btnStart.onclick=async ()=>{
    if(client){ try{client.disconnect();}catch(e){} }
    transcriptsLog.innerHTML=''; eventsBody.innerHTML=''; rawLog.innerHTML=''; rawEvents=[]; eventCounter=0; finalCount=0; partialCount=0; ttsCount=0; refreshStatus(); ingestModeBadge.textContent='mode: -'; connState.textContent='init';
    client=new StreamingVoiceClient({language:'en', debug:true, onEvent:handleEvent});
    startTs=Date.now();
    try{ client.connect(); await client.start(); sampleRateBadge.textContent='sr: '+(client.audioContext?client.audioContext.sampleRate:'?'); connState.textContent='starting'; btnSuspend.disabled=false; btnDisconnect.disabled=false; }catch(e){ connState.textContent='start_error'; logRaw({evt:'start_error', error:String(e)}); }
};
btnSuspend.onclick=()=>{ if(!client) return; client.setSuspended(true); connState.textContent='suspended'; btnSuspend.disabled=true; btnResume.disabled=false; };
btnResume.onclick=()=>{ if(!client) return; client.setSuspended(false); connState.textContent='resuming'; btnSuspend.disabled=false; btnResume.disabled=true; };
btnDisconnect.onclick=()=>{ if(!client) return; client.disconnect(); connState.textContent='disconnecting'; };
btnClear.onclick=()=>{ rawLog.innerHTML=''; rawEvents=[]; };
exportLink.onclick=(e)=>{ e.preventDefault(); if(!rawEvents.length) return; const report={ generated_at:new Date().toISOString(), total_events:rawEvents.length, partials:partialCount, finals:finalCount, tts:ttsCount, events:rawEvents }; const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='streaming_debug_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json'; document.body.appendChild(a); a.click(); setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 800); };
</script>
</body></html>'''
        return Response(html, mimetype='text/html')

@app.route('/worklet/pcm_downsampler.js')
def serve_worklet_pcm_downsampler():
    js = r'''class PcmDownsampler extends AudioWorkletProcessor { constructor(opts){ super(); const o=opts.processorOptions||{}; this.target=o.targetSampleRate||16000; this.frame=o.frameSamples||320; this.srcRate=sampleRate; this.ratio=this.srcRate/this.target; this.buffer=new Float32Array(0);} process(inputs){ const ch=inputs[0]; if(!ch||ch.length===0) return true; const data=ch[0]; // append
        const merged=new Float32Array(this.buffer.length+data.length); merged.set(this.buffer,0); merged.set(data,this.buffer.length); this.buffer=merged; const need=this.frame*this.ratio; if(this.buffer.length<need) return true; // produce one frame
        const out=new Int16Array(this.frame); for(let i=0;i<this.frame;i++){ const srcPos=i*this.ratio; const i0=Math.floor(srcPos); const i1=Math.min(i0+1,this.buffer.length-1); const frac=srcPos-i0; const sample=this.buffer[i0]+(this.buffer[i1]-this.buffer[i0])*frac; let s=Math.max(-1,Math.min(1,sample)); out[i]=s<0? s*0x8000 : s*0x7FFF; }
        // consume
        const consumed=Math.floor(this.frame*this.ratio); this.buffer=this.buffer.slice(consumed); this.port.postMessage(out.buffer,[out.buffer]); return true; }} registerProcessor('pcm-downsampler',PcmDownsampler);'''
    return Response(js, mimetype='application/javascript')

if __name__ == '__main__':
    import ssl
    from pathlib import Path
    
    # Allow port configuration via environment variable
    port = int(os.environ.get('PORT', 5001))
    
    # Check if running in production mode
    is_production = os.environ.get('FLASK_ENV') == 'production'
    debug_mode = not is_production
    
    # Check if we're running behind a proxy (like Nginx)
    behind_proxy = os.environ.get('BEHIND_PROXY', 'false').lower() == 'true'
    
    if behind_proxy or is_production:
        # When behind a proxy (like Nginx), run HTTP only
        # The proxy handles SSL termination
        if not is_production:
            print("🔄 Running behind proxy - using HTTP")
            print(f"📱 Proxy should forward to: http://localhost:{port}")
            print("🔒 SSL/HTTPS handled by Nginx proxy")
        app.run(host='127.0.0.1', port=port, debug=debug_mode)
    else:
        # Development mode - run HTTP only (SSL handled by Nginx)
        if not is_production:
            print("� Development mode - using HTTP")
            print(f"📱 Access at: http://localhost:{port}")
            print("🔒 For production, use Nginx with Let's Encrypt SSL")
        app.run(host='127.0.0.1', port=port, debug=debug_mode)
