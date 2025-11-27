# Developer Mode: Disabling Kesay Clinics Safeguards 🔓

## Overview

The Kesay Clinics system prompt safeguards can be **temporarily disabled** for development and testing purposes. This allows you to:

- ✅ Chat about any topic (not just clinic-related)
- ✅ Get responses in any language (not just Arabic)
- ✅ Test voice-to-voice conversations freely
- ✅ Develop and debug without restrictions

⚠️ **IMPORTANT**: This is for **TESTING ONLY**. Never disable safeguards in production with real patients!

---

## Quick Start

### Method 1: Shell Script (Easiest) 🚀

```bash
# Disable safeguards
cd /home/lumi/beautyai/backend
./toggle_safeguards.sh disable

# Check status
./toggle_safeguards.sh status

# Re-enable safeguards
./toggle_safeguards.sh enable
```

### Method 2: Environment Variable 🌍

```bash
# Disable safeguards
export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1

# Restart API service
sudo systemctl restart beautyai-api.service

# Re-enable (unset variable)
unset DISABLE_SYSTEM_PROMPT_SAFEGUARDS
sudo systemctl restart beautyai-api.service
```

### Method 3: Per-Request API Flag 🎯

Add to your API request:

```json
{
  "model_name": "qwen3-unsloth-q4ks",
  "message": "Tell me about the weather",
  "generation_config": {
    "disable_system_prompt_safeguards": true
  }
}
```

### Method 4: Python Code 🐍

```python
from dev_safeguard_helper import with_safeguards_disabled

@with_safeguards_disabled
def test_unrestricted_chat():
    # Your test code here
    pass
```

---

## Detailed Methods

### 1. Shell Script Method (Recommended for Quick Testing)

#### Disable Safeguards

```bash
cd /home/lumi/beautyai/backend
./toggle_safeguards.sh disable
```

Output:
```
🔓 Safeguards DISABLED (Developer Mode)
   Model will use default medical prompts
   No topic restrictions
   Language detection works normally

⚠️  WARNING: This is for TESTING ONLY!
   DO NOT use in production with real patients

💡 To apply changes:
   1. Source the env file: source /home/lumi/beautyai/backend/.env.safeguards
   2. Restart API: sudo systemctl restart beautyai-api.service
```

#### Check Status

```bash
./toggle_safeguards.sh status
```

#### Re-enable Safeguards

```bash
./toggle_safeguards.sh enable
```

---

### 2. Environment Variable Method (For Service-Wide Disable)

#### For systemd Service

Edit the service file:

```bash
sudo nano /etc/systemd/system/beautyai-api.service
```

Add environment variable to `[Service]` section:

```ini
[Service]
Environment="DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1"
```

Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api.service
```

#### For Direct Python Execution

```bash
export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1
cd /home/lumi/beautyai && python backend/run_server.py
```

#### For VS Code Task (Direct Mode)

Edit task in `.vscode/tasks.json`:

```json
{
  "label": "🔥 Dev: Run API (NO SAFEGUARDS)",
  "type": "shell",
  "command": "cd /home/lumi/beautyai && source backend/venv/bin/activate && export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1 && python backend/run_server.py"
}
```

---

### 3. Per-Request API Flag (For Specific Tests)

#### Chat API Example

```python
import requests

response = requests.post('http://localhost:8000/api/chat', json={
    "model_name": "qwen3-unsloth-q4ks",
    "message": "What's the weather today?",
    "generation_config": {
        "max_tokens": 200,
        "temperature": 0.7,
        "disable_system_prompt_safeguards": True  # 🔓 Bypass
    }
})

print(response.json())
```

#### WebSocket Voice Example

```python
import asyncio
import websockets
import json

async def test_voice_without_safeguards():
    uri = "ws://localhost:8000/api/streaming-voice?language=ar"
    
    async with websockets.connect(uri) as websocket:
        # Send config message
        await websocket.send(json.dumps({
            "type": "config",
            "generation_config": {
                "disable_system_prompt_safeguards": True
            }
        }))
        
        # Continue with voice testing...
        pass

asyncio.run(test_voice_without_safeguards())
```

---

### 4. Python Helper Module

The `dev_safeguard_helper.py` module provides utilities:

```python
from dev_safeguard_helper import (
    set_safeguards_disabled,
    are_safeguards_disabled,
    with_safeguards_disabled,
    add_safeguard_bypass_to_config
)

# Check current status
if are_safeguards_disabled():
    print("Safeguards are OFF")

# Temporarily disable
set_safeguards_disabled(True)

# Use decorator
@with_safeguards_disabled
def my_test():
    # Runs without safeguards
    pass

# Add to config
config = {"temperature": 0.7}
config = add_safeguard_bypass_to_config(config)
```

---

## Use Cases

### Voice-to-Voice Testing 🎤

```bash
# 1. Disable safeguards
export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1
sudo systemctl restart beautyai-api.service

# 2. Run voice tests
cd /home/lumi/beautyai
python tests/streaming/ws_replay_pcm.py --file voice_tests/input_test_questions/pcm/q1.pcm --language ar

# 3. Re-enable after testing
unset DISABLE_SYSTEM_PROMPT_SAFEGUARDS
sudo systemctl restart beautyai-api.service
```

### General Conversation Testing 💬

```python
# test_general_chat.py
from dev_safeguard_helper import with_safeguards_disabled
from beautyai_inference.services.inference.chat_service import ChatService
from beautyai_inference.config.config_manager import AppConfig

@with_safeguards_disabled
def test_general_topics():
    chat = ChatService()
    chat.load_default_model_from_config()
    
    # Test weather question
    response, lang, _, _ = chat.chat(
        message="What's the weather today?",
        model_name="qwen3-unsloth-q4ks",
        model_config=chat.default_model_config,
        generation_config={"max_tokens": 200}
    )
    
    print(f"Response: {response}")
    print(f"Language: {lang}")

test_general_topics()
```

### API Integration Testing 🔗

```python
# test_api_bypass.py
import requests

def test_chat_without_restrictions():
    response = requests.post('http://localhost:8000/api/chat', json={
        "message": "Tell me a joke",
        "generation_config": {
            "disable_system_prompt_safeguards": True
        }
    })
    
    assert response.status_code == 200
    data = response.json()
    print(f"Response: {data['response']}")
    # Should respond with a joke, not refuse

test_chat_without_restrictions()
```

---

## Verification

### Check if Safeguards are Disabled

#### In Logs

When safeguards are disabled, you'll see:

```
🔓 DEVELOPER MODE: System prompt safeguards DISABLED (testing only)
[prompt] 🔄 Reset to default system prompts (safeguards removed)
```

When enabled:

```
✅ Applied model-specific system prompt for language: ar
```

#### Via API Test

```bash
# With safeguards DISABLED - should get response about weather
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather today?",
    "generation_config": {
      "disable_system_prompt_safeguards": true
    }
  }'

# With safeguards ENABLED - should refuse and redirect to clinic topics
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather today?"
  }'
```

---

## Behavior Comparison

| Feature | Safeguards ON 🔒 | Safeguards OFF 🔓 |
|---------|------------------|-------------------|
| **Topics** | Only Kesay Clinics | Any topic |
| **Language** | Arabic ONLY | Auto-detected |
| **Doctors** | Must mention | Not required |
| **Off-topic** | Politely refuses | Answers freely |
| **System Prompt** | Kesay Clinics instructions | Default medical prompts |

---

## Best Practices

### ✅ DO:
- Use safeguard bypass **only** for development/testing
- Document when you disable safeguards
- Re-enable immediately after testing
- Use per-request flag for specific tests
- Check logs to confirm bypass is working

### ❌ DON'T:
- Deploy with safeguards disabled to production
- Leave `DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1` in service file
- Use bypass with real patient data
- Forget to re-enable after testing
- Share API with bypass flag enabled

---

## Troubleshooting

### Safeguards Not Disabling

1. **Check environment variable:**
   ```bash
   echo $DISABLE_SYSTEM_PROMPT_SAFEGUARDS
   # Should output: 1
   ```

2. **Verify service restart:**
   ```bash
   sudo systemctl status beautyai-api.service
   # Check if service restarted recently
   ```

3. **Check logs:**
   ```bash
   sudo journalctl -u beautyai-api.service -n 50 | grep -i safeguard
   # Look for: "DEVELOPER MODE: System prompt safeguards DISABLED"
   ```

### Safeguards Not Re-enabling

1. **Unset variable:**
   ```bash
   unset DISABLE_SYSTEM_PROMPT_SAFEGUARDS
   ```

2. **Remove .env file:**
   ```bash
   rm /home/lumi/beautyai/backend/.env.safeguards
   ```

3. **Restart service:**
   ```bash
   sudo systemctl restart beautyai-api.service
   ```

4. **Verify in logs:**
   ```bash
   sudo journalctl -u beautyai-api.service -n 50 | grep -i "system prompt"
   # Look for: "Applied model-specific system prompt"
   ```

---

## Files Reference

### Created Files

1. **`backend/toggle_safeguards.sh`** - Shell script to enable/disable
2. **`backend/dev_safeguard_helper.py`** - Python helper utilities
3. **`docs/DEVELOPER_MODE_SAFEGUARDS.md`** - This documentation

### Modified Files

1. **`backend/src/beautyai_inference/services/inference/chat_service.py`**
   - Added safeguard bypass logic
   - Checks environment variable and generation_config flag

2. **`backend/src/beautyai_inference/services/voice/conversation/simple_voice_service.py`**
   - Propagates bypass flag to generation config
   - Supports environment variable

3. **`backend/src/beautyai_inference/services/shared/prompt_building_service.py`**
   - Added `reset_to_default_prompts()` method
   - Stores backup of default prompts

---

## Security Notes

🔐 **This feature is designed for YOUR testing only!**

- The bypass flag is **not** exposed in production APIs
- Requires direct access to environment variables or code
- Logs clearly indicate when safeguards are disabled
- No user-facing UI for disabling safeguards

---

## Quick Reference Card

```bash
# ═══════════════════════════════════════════════════
#  KESAY CLINICS SAFEGUARD CONTROLS - QUICK REF
# ═══════════════════════════════════════════════════

# STATUS CHECK
./backend/toggle_safeguards.sh status

# DISABLE (for testing)
./backend/toggle_safeguards.sh disable
source backend/.env.safeguards
sudo systemctl restart beautyai-api.service

# ENABLE (back to production)
./backend/toggle_safeguards.sh enable
sudo systemctl restart beautyai-api.service

# PER-REQUEST BYPASS (API)
{
  "generation_config": {
    "disable_system_prompt_safeguards": true
  }
}

# VERIFY IN LOGS
sudo journalctl -u beautyai-api.service -f | grep safeguard
```

---

**Created by**: Lumina Ashley 💕  
**Date**: November 26, 2025  
**Purpose**: Development & Testing Only
