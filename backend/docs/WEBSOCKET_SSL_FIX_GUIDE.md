# 🔧 WebSocket SSL/HTTPS Configuration Guide

## Problem Summary
- **Issue**: WebSocket connections from HTTPS pages fail due to mixed content security
- **Current**: HTTPS page trying to connect to `ws://dev.gmai.sa:8000` (insecure)
- **Solution**: Use nginx proxy to provide `wss://dev.gmai.sa/ws/voice-conversation` (secure)

## ✅ Step-by-Step Fix

### Step 1: Update Nginx Configuration

1. **Copy the updated config file:**
```bash
sudo cp /home/lumi/benchmark_and_test/webapp-https /etc/nginx/sites-available/dev.gmai.sa
```

2. **Test nginx configuration:**
```bash
sudo nginx -t
```

3. **If test passes, reload nginx:**
```bash
sudo systemctl reload nginx
```

### Step 2: Verify BeautyAI API is Running

```bash
# Check if BeautyAI API is running on port 8000
sudo systemctl status beautyai-api

# If not running, start it
sudo systemctl start beautyai-api

# Check if port 8000 is listening
sudo netstat -tlnp | grep :8000
```

### Step 3: Test the Configuration

```bash
cd /home/lumi/beautyai
python test_wss_proxy.py
```

### Step 4: Update Your Web Application

The WebSocket URLs will now be:
- **HTTPS pages**: `wss://dev.gmai.sa/ws/voice-conversation` ✅
- **HTTP pages**: `ws://dev.gmai.sa/ws/voice-conversation` ✅

## 🎯 What Changed in Nginx Config

### New Proxy Routes Added:

1. **WebSocket Route**: `/ws/` → `http://localhost:8000`
   - Handles WebSocket upgrade properly
   - Long timeouts for persistent connections
   - SSL termination at nginx level

2. **API Routes**: `/inference/`, `/health`, `/docs`, `/models` → `http://localhost:8000`
   - Handles BeautyAI API calls
   - Large file upload support (100MB for audio files)
   - Proper SSL headers

3. **Main App Route**: `/` → `http://localhost:5000`
   - Your existing web application
   - Unchanged behavior

## 🌐 URL Mapping After Configuration

| Original URL | New Proxied URL | Purpose |
|--------------|-----------------|---------|
| `ws://dev.gmai.sa:8000/ws/voice-conversation` | `wss://dev.gmai.sa/ws/voice-conversation` | WebSocket voice chat |
| `http://dev.gmai.sa:8000/inference/voice-to-voice` | `https://dev.gmai.sa/inference/voice-to-voice` | REST API calls |
| `http://dev.gmai.sa:8000/health` | `https://dev.gmai.sa/health` | Health checks |
| `http://dev.gmai.sa:8000/docs` | `https://dev.gmai.sa/docs` | API documentation |

## 🚀 Benefits After Fix

1. **✅ Mixed Content Resolved**: HTTPS pages can connect to WSS WebSockets
2. **✅ Single Domain**: All services accessible through `dev.gmai.sa`
3. **✅ SSL Everywhere**: All communication encrypted
4. **✅ No Port Exposure**: Users don't need to know about port 8000
5. **✅ Simplified URLs**: Clean, professional URLs

## 🧪 Testing Your Voice Chat

After applying the nginx config:

1. **Open your web app**: `https://dev.gmai.sa/`
2. **Navigate to voice chat page**
3. **Click "Connect to BeautyAI"**
4. **You should see**: "Connected to BeautyAI WebSocket!" ✅
5. **Start voice conversation!**

## 🔧 Troubleshooting

### If WSS connection fails:
1. Check nginx error logs: `sudo tail -f /var/log/nginx/error.log`
2. Check BeautyAI API logs: `sudo journalctl -u beautyai-api -f`
3. Verify port 8000 is listening: `sudo netstat -tlnp | grep :8000`

### If API calls fail:
1. Test direct API: `curl http://localhost:8000/health`
2. Test through proxy: `curl https://dev.gmai.sa/health`
3. Check nginx access logs: `sudo tail -f /var/log/nginx/access.log`

## ✅ Success Indicators

After successful configuration:
- ✅ `python test_wss_proxy.py` shows WSS connection working
- ✅ Browser console shows no Mixed Content errors
- ✅ Voice conversations stream in real-time (no file downloads)
- ✅ All API calls work through HTTPS

## 🎉 Final Result

Your voice conversation system will work seamlessly:
- **Real-time streaming**: No more file downloads
- **Secure connections**: Full HTTPS/WSS encryption
- **Professional URLs**: Clean `dev.gmai.sa` domain
- **Browser compatibility**: Works on all modern browsers
