# 🚀 Quick Start Guide - BeautyAI PABX

Get your PABX system running in 10 minutes!

---

## Prerequisites Check

```bash
# Check Python 3.8+ is installed
python3 --version

# Check Node.js 16+ for frontend
node --version

# If missing, install:
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nodejs npm portaudio19-dev
```

---

## Step 1: Install Backend Dependencies

```bash
cd /home/lumi/beautyai/pabx

# Option A: Use install script (recommended)
./install.sh

# Option B: Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Answer 'y' when prompted to install system dependencies and systemd services.

## Step 2: Install Frontend Dependencies

```bash
# Navigate to UI directory
cd /home/lumi/beautyai/pabx/ui

# Install Node.js packages
npm install

# Verify installation
npm list react typescript vite
```

This installs React 18, TypeScript, Vite, and all UI dependencies.

---

## Step 3: Configure HT813 Device

Edit `config/settings.yaml`:
```yaml
capture:
  target_ip: "192.168.100.96"  # Your HT813 IP

ht813:
  ip_address: "192.168.100.96"
  username: "admin"
  password: "admin"
```

**Find your HT813 IP:**
```bash
# Check your router's DHCP leases
# Or use network scanner
sudo nmap -sn 192.168.100.0/24 | grep -B 2 "Grandstream"
```

---

## Step 4: Start the Backend Server

### Option A: Development Mode (Terminal 1)
```bash
cd /home/lumi/beautyai/pabx
source venv/bin/activate
./run_server.py --mode api
```

**Expected output:**
```
🚀 BeautyAI PABX Server Starting...
📋 Mode: API Server
🌐 SIP Server listening on 0.0.0.0:5060
📊 RTP Handler initialized
🎯 API Server: http://0.0.0.0:8080
🔌 WebSocket: ws://0.0.0.0:8080/ws
✅ Server running!
```

### Option B: Systemd Service (Production)
```bash
sudo systemctl start pabx-api
sudo systemctl status pabx-api
```

---

## Step 5: Start the Frontend (Terminal 2)

```bash
cd /home/lumi/beautyai/pabx/ui
npm run dev
```

**Expected output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.100.x:3000/
```

**Open in browser:** http://localhost:3000

You should see the PABX Web UI with:
- 📞 **Call List** (left top)
- 📊 **Call Details** (right top)
- 📱 **Device Dashboard** (left bottom)
- 📡 **Capture Status** (right bottom)

---

## Step 6: Verify System is Running

### Check API Health
```bash
curl http://localhost:8080/api/health
# Expected: {"status":"ok","version":"1.0.0"}
```

### Check Web UI
Open browser at http://localhost:3000
- ✅ Connection indicator should be **green** (top-right)
- ✅ "No active calls" message in Call List
- ✅ WebSocket connected (check browser console)

### Check Backend Logs
```bash
tail -f logs/api_server.log
```

---

## Step 7: Configure Your HT813

### Access HT813 Web Interface
1. Open: `http://192.168.100.96`
2. Login: `admin` / `admin`

### Configure FXS Port 1

**Navigate to:** FXS PORT 1 → SIP Settings

**Required settings:**
```
Primary SIP Server: <YOUR_COMPUTER_IP>:5060
  Example: 192.168.100.50:5060
  
SIP User ID: 1001
Authenticate ID: 1001  
Authenticate Password: password123

Preferred Vocoder: choice 0 = PCMU
Enable Silence Suppression: No
Enable Call Features: Yes
```

**Find your computer's IP:**
```bash
ip addr show | grep "inet " | grep -v 127.0.0.1
# Or use: hostname -I
```

3. Click **Apply** and wait 10 seconds
4. Check **Status** page for "Registered"

---

## Step 8: Test the System!

### Test 1: Check Registration

**In Web UI (Device Dashboard):**
- Should show HT813 IP, MAC, firmware
- FXS Port 1: "Registered" badge (green)

**In Backend Terminal:**
```
📞 SIP REGISTER received from 192.168.100.96:5060
✅ Registration accepted for 1001
```

### Test 2: Make Your First Call! 📞

1. **Pick up analog phone** connected to HT813 FXS Port 1
2. **Dial any number** (e.g., 555-1234)
3. **Watch the magic happen!** ✨

**What you should see:**

**In Web UI:**
- 🟢 Call appears in **Call List** with "Ringing" badge
- 🔔 Connection indicator pulses
- ⏱️ Time shows "a few seconds ago"

**After 3 seconds (auto-answer):**
- 📞 Badge changes to "Active" (green)
- 📊 **Call Details** shows RTP statistics
- 📈 Packets sent/received counters increase
- 🎵 You hear audio confirmation

**In Backend Terminal:**
```
📞 INVITE received from 1001
✅ Call answered automatically
🎵 RTP stream established
📊 Codec: PCMU, Rate: 8000 Hz
```

4. **Hang up the phone**
   - Badge changes to "Ended"
   - Call stays in list for 30 seconds
   - Recording saved (if enabled)

### Test 3: Verify Real-Time Updates

**Open Browser Console** (F12 → Console):
```javascript
WebSocket connected: ws://localhost:8080/ws
Received event: {"type":"call_incoming","data":{...}}
Received event: {"type":"call_answered","data":{...}}
```

---

## Step 9: Advanced Testing

### Enable Packet Capture
```bash
# Start packet capture
curl -X POST http://localhost:8080/api/capture/start

# Check status
curl http://localhost:8080/api/capture/status
```

**In Web UI (Capture Status):**
- Status changes to "Enabled"
- Packet counts update in real-time
- SIP/RTP/RTCP breakdown shows distribution

### Test Call Recording
```bash
# Get active call ID from Web UI or:
CALL_ID=$(curl -s http://localhost:8080/api/calls | jq -r '.[0].id')

# Start recording
curl -X POST http://localhost:8080/api/calls/$CALL_ID/record/start

# Make some conversation...

# Stop recording  
curl -X POST http://localhost:8080/api/calls/$CALL_ID/record/stop

# Find recording
ls -la recordings/$CALL_ID/
```

### Test Device Reboot
**In Web UI:**
1. Go to **Device Dashboard**
2. Click **Reboot Device** button
3. Confirm in dialog
4. HT813 reboots (~30 seconds)
5. Dashboard shows "Offline" then "Online" again

### Configure Second FXS Port
Repeat Step 7 for **FXS PORT 2**:
- SIP User ID: `1002`
- Make simultaneous calls from both ports
- Watch both in Call List!

---

## Troubleshooting

## Troubleshooting

### ❌ Backend Won't Start

**Error: "Address already in use"**
```bash
# Find process using port 8080
sudo lsof -i :8080

# Kill it
sudo kill -9 <PID>

# Or use different port
./run_server.py --mode api --api-port 8081
```

**Error: "Permission denied" (port 5060)**
```bash
# Use non-privileged port
./run_server.py --mode api --sip-port 5160

# Update HT813 to use :5160
```

**Error: "PyAudio not found"**
```bash
sudo apt install portaudio19-dev
source venv/bin/activate
pip install --force-reinstall pyaudio
```

### ❌ Frontend Won't Start

**Error: "Cannot find module"**
```bash
cd ui
rm -rf node_modules package-lock.json
npm install
```

**Error: "Port 3000 already in use"**
```bash
npm run dev -- --port 3001
```

**Error: "Failed to fetch" in browser**
```bash
# Check backend is running
curl http://localhost:8080/api/health

# Check vite.config.ts proxy settings
cat ui/vite.config.ts | grep proxy
```

### ❌ HT813 Won't Register

**Step 1: Check Network**
```bash
# Can you ping HT813?
ping 192.168.100.96

# Can you reach web interface?
curl -I http://192.168.100.96
```

**Step 2: Check HT813 Config**
- Primary SIP Server = `<YOUR_IP>:5060` (not 192.168.100.96!)
- Find your IP: `hostname -I | awk '{print $1}'`
- Example: `192.168.100.50:5060`

**Step 3: Check Firewall**
```bash
# Allow SIP port
sudo ufw allow 5060/udp

# Allow RTP ports  
sudo ufw allow 10000:20000/udp

# Check if listening
sudo netstat -ulnp | grep 5060
```

**Step 4: Check Backend Logs**
```bash
tail -f logs/sip_server.log | grep REGISTER
```

Should see:
```
📞 SIP REGISTER received from 192.168.100.96
```

If not, HT813 isn't sending REGISTER messages.

### ❌ WebSocket Not Connecting

**Check browser console (F12):**
```
Error: WebSocket connection failed
```

**Fixes:**
1. Verify backend running: `curl http://localhost:8080/api/health`
2. Check proxy in `ui/vite.config.ts`
3. Hard refresh: **Ctrl+F5**
4. Check firewall isn't blocking WebSocket

### ❌ No Audio / One-Way Audio

**Check RTP ports:**
```bash
# Open RTP port range
sudo ufw allow 10000:20000/udp

# Verify ports available
sudo netstat -ulnp | grep -E "1[0-9]{4}"
```

**Check PyAudio:**
```bash
source venv/bin/activate
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Devices: {p.get_device_count()}')"
```

**Check codec:**
- HT813 must use PCMU or PCMA
- Check logs for: `Codec: PCMU`

### ❌ Calls Don't Appear in UI

**Check WebSocket in browser console:**
```javascript
// Should see:
WebSocket connected: ws://localhost:8080/ws

// Make a call, should see:
Received event: {type: "call_incoming", data: {...}}
```

**If no events:**
1. Backend not sending events
2. Check logs: `tail -f logs/api_server.log`
3. Restart backend

**If events received but UI doesn't update:**
1. Check browser console for errors
2. State management issue - refresh page

---

## Common Commands Reference

### Backend Control
```bash
# Start backend
cd /home/lumi/beautyai/pabx
source venv/bin/activate
./run_server.py --mode api

# With debug logging
./run_server.py --mode api --log-level DEBUG

# Different ports
./run_server.py --mode api --sip-port 5160 --api-port 8081

# View logs
tail -f logs/api_server.log
tail -f logs/sip_server.log  
tail -f logs/rtp_handler.log
```

### Frontend Control
```bash
# Start dev server
cd /home/lumi/beautyai/pabx/ui
npm run dev

# Different port
npm run dev -- --port 3001

# Build for production
npm run build

# Preview production build
npm run preview
```

### API Testing
```bash
# Health check
curl http://localhost:8080/api/health

# List calls
curl http://localhost:8080/api/calls

# HT813 status
curl http://localhost:8080/api/ht813/status

# HT813 statistics
curl http://localhost:8080/api/ht813/statistics

# Start capture
curl -X POST http://localhost:8080/api/capture/start

# Stop capture
curl -X POST http://localhost:8080/api/capture/stop

# Capture status
curl http://localhost:8080/api/capture/status

# Reboot HT813
curl -X POST http://localhost:8080/api/ht813/reboot
```

### System Management
```bash
# Start as service
sudo systemctl start pabx-api

# Stop service
sudo systemctl stop pabx-api

# Check status
sudo systemctl status pabx-api

# View service logs
sudo journalctl -u pabx-api -f

# Enable on boot
sudo systemctl enable pabx-api
```

---

## Production Deployment

### Build Frontend
```bash
cd ui
npm run build
# Output: dist/ directory
```

### Install as Service
```bash
# Copy service file
sudo cp systemd/pabx-api.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/pabx-api.service

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable pabx-api
sudo systemctl start pabx-api
```

### Serve Frontend with Backend
Backend automatically serves `ui/dist/` at root path when built.

### Optional: Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name pabx.example.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
    }
    
    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Quick Testing Checklist

### ✅ Backend Running
```bash
curl http://localhost:8080/api/health
# Should return: {"status":"ok","version":"1.0.0"}
```

### ✅ Frontend Running
Open http://localhost:3000
- Connection indicator is green
- No console errors (F12)

### ✅ HT813 Configured
- Web interface accessible: http://192.168.100.96
- Primary SIP Server = YOUR_IP:5060
- Status page shows "Registered"

### ✅ Call Works
- Pick up phone
- Dial any number
- Call appears in Web UI
- RTP statistics show packets

### ✅ WebSocket Works
Browser console shows:
```
WebSocket connected
Received event: {type: "call_incoming"}
```

---

## Important URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Web UI |
| **Backend API** | http://localhost:8080 | REST API |
| **API Docs** | http://localhost:8080/docs | Swagger UI |
| **WebSocket** | ws://localhost:8080/ws | Real-time events |
| **HT813 Web** | http://192.168.100.96 | Device config |

---

## File Locations

```
/home/lumi/beautyai/pabx/
├── ui/                    # React frontend
│   ├── src/              # Source code
│   ├── dist/             # Production build
│   └── node_modules/     # Dependencies
├── src/                   # Backend source
├── config/               # Configuration
│   └── settings.yaml     # Main config
├── logs/                 # Log files
│   ├── api_server.log
│   ├── sip_server.log
│   └── rtp_handler.log
├── recordings/           # Call recordings
├── captures/             # Packet captures
├── venv/                 # Python virtual env
└── run_server.py        # Entry point
```

---

## Next Steps After Testing

1. **✅ Verified basic calls work?**
   - Try call recording
   - Enable packet capture
   - Test with second FXS port

2. **✅ Ready for production?**
   - Build frontend: `npm run build`
   - Install systemd service
   - Configure firewall rules
   - Set up monitoring

3. **✅ Want more features?**
   - Read full docs: `README.md`
   - Explore API: http://localhost:8080/docs
   - Check Phase 6 (CLI tool)
   - Check Phase 7 (test suite)

---

## Getting Help

### Documentation
- **Main README**: `/home/lumi/beautyai/pabx/README.md`
- **Frontend Guide**: `/home/lumi/beautyai/pabx/ui/FRONTEND_GUIDE.md`
- **API Reference**: http://localhost:8080/docs

### Logs
- Backend: `logs/api_server.log`, `logs/sip_server.log`
- Frontend: Browser console (F12)
- System: `sudo journalctl -u pabx-api`

### Debug Mode
```bash
./run_server.py --mode api --log-level DEBUG
```

---

**🎉 That's it! You're ready to make VoIP calls!**

**Questions?** Check the logs, API docs, or full README.

**Happy Testing!** 📞✨

## Common Commands

```bash
# View all logs
tail -f logs/system/app.json | jq .

# View systemd logs
sudo journalctl -u pabx-server -f

# Restart service
sudo systemctl restart pabx-server

# Stop all services
sudo systemctl stop pabx-server pabx-sniffer

# Check Python virtual environment
source venv/bin/activate
pip list | grep -E "fastapi|pyaudio|scapy"
```

## Next Steps

- Read full documentation: `README.md`
- Configure auto-answer: edit `config/settings.yaml`
- Set up call recording
- Explore API endpoints at `/docs`
- Build React frontend (Phase 5)

## Support

For issues, check:
1. System logs: `logs/system/app.json`
2. Systemd logs: `sudo journalctl -u pabx-server`
3. Session traces: `logs/sessions/`
4. Network capture: `captures/`
