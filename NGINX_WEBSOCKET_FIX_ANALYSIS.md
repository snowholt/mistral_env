# 🔧 Nginx WebSocket Configuration Fix Analysis

## 📋 Issue Summary

**Problem**: WebSocket endpoints for BeautyAI SimpleVoiceService are returning HTTP 502 errors due to incorrect nginx proxy configuration.

**Root Cause**: The deployed nginx configuration at `/etc/nginx/sites-enabled/gmai.sa` has incorrect `proxy_pass` directives that strip the WebSocket path prefix.

## 🔍 Detailed Analysis

### Current (Broken) Configuration
```nginx
location /api/v1/ws/ {
    proxy_pass http://localhost:8000;  # ❌ INCORRECT - missing path
}

location /ws/ {
    proxy_pass http://localhost:8000;  # ❌ INCORRECT - missing path  
}
```

### Expected Request Flow (Current - Broken)
1. Client connects to: `wss://api.gmai.sa/api/v1/ws/simple-voice-chat`
2. Nginx location `/api/v1/ws/` matches
3. Nginx strips `/api/v1/ws/` and forwards to: `http://localhost:8000/simple-voice-chat` ❌
4. FastAPI receives: `/simple-voice-chat` (which doesn't exist)
5. Result: HTTP 404/502 error

### Fixed Configuration
```nginx
location /api/v1/ws/ {
    proxy_pass http://localhost:8000/api/v1/ws/;  # ✅ CORRECT - preserves path
}

location /ws/ {
    proxy_pass http://localhost:8000/ws/;  # ✅ CORRECT - preserves path
}
```

### Expected Request Flow (Fixed)
1. Client connects to: `wss://api.gmai.sa/api/v1/ws/simple-voice-chat`
2. Nginx location `/api/v1/ws/` matches  
3. Nginx forwards to: `http://localhost:8000/api/v1/ws/simple-voice-chat` ✅
4. FastAPI receives: `/api/v1/ws/simple-voice-chat` (which exists)
5. Result: Successful WebSocket connection

## 🛠️ Fix Implementation

The fix requires updating the deployed nginx configuration with the correct `proxy_pass` directives.

### Required Changes

**File**: `/etc/nginx/sites-enabled/gmai.sa`

**Change 1**: WebSocket Endpoints
```diff
location /api/v1/ws/ {
-   proxy_pass http://localhost:8000;
+   proxy_pass http://localhost:8000/api/v1/ws/;
    # ... rest of configuration unchanged
}
```

**Change 2**: Legacy WebSocket Endpoints  
```diff
location /ws/ {
-   proxy_pass http://localhost:8000;
+   proxy_pass http://localhost:8000/ws/;
    # ... rest of configuration unchanged
}
```

## 📋 Deployment Checklist

- [ ] **Backup current configuration**
  ```bash
  sudo cp /etc/nginx/sites-enabled/gmai.sa /etc/nginx/sites-enabled/gmai.sa.backup.$(date +%Y%m%d_%H%M%S)
  ```

- [ ] **Apply the configuration fix**
  ```bash
  sudo cp /path/to/fixed/nginx-clean-config.conf /etc/nginx/sites-enabled/gmai.sa
  ```

- [ ] **Test nginx configuration syntax**
  ```bash
  sudo nginx -t
  ```

- [ ] **Reload nginx configuration**  
  ```bash
  sudo systemctl reload nginx
  ```

- [ ] **Verify WebSocket connectivity**
  ```bash
  python test_websocket_nginx_fix.py
  ```

## 🧪 Testing

The `test_websocket_nginx_fix.py` script will validate:

1. **Primary WebSocket Path**: `/api/v1/ws/simple-voice-chat`
2. **Legacy WebSocket Path**: `/ws/simple-voice-chat`  
3. **HTTP Status Endpoint**: `/api/v1/ws/simple-voice-chat/status`

Expected results after fix:
- WebSocket connections should succeed (or get proper WebSocket-level errors, not HTTP 502)
- HTTP endpoints should return proper JSON responses (not nginx error pages)

## 🚨 Important Notes

1. **Backend Service Required**: The FastAPI backend service must be running on `localhost:8000` for the fix to work completely.

2. **WebSocket Router Registration**: Ensure the `websocket_simple_voice_router` is properly registered in the FastAPI app:
   ```python
   app.include_router(websocket_simple_voice_router, prefix="/api/v1", tags=["simple-voice"])
   ```

3. **Service Dependencies**: The SimpleVoiceService requires proper initialization of:
   - Edge TTS service
   - Whisper transcription service  
   - Chat/inference service

## ✅ Validation

After applying the fix, you should see:
- HTTP 502 errors eliminated
- WebSocket upgrade requests properly forwarded
- Successful connection to WebSocket endpoints (assuming backend is running)

The HTTP 502 errors were nginx's way of saying "I forwarded the request but got no response", which happened because the requests were being forwarded to non-existent paths due to the missing path prefix in `proxy_pass`.
