# HT813 Audio Capture - Complete Setup Guide

## 🚨 Current Problem

**The HT813 cannot register because there's NO SIP SERVER running on 192.168.100.39!**

Both ports show: `Not Registered`

## ✅ Solution: Run a Test SIP Server

### Step 1: Start the SIP Server

Open a terminal and run:

```bash
cd /home/lumi/beautyai/pabx
sudo python3 simple_sip_server.py
```

You should see:
```
🎙️  Simple SIP Server Started
   Listening on: 0.0.0.0:5060
   Press Ctrl+C to stop
```

**Keep this terminal open!**

### Step 2: Fix HT813 Settings

Access the HT813 web interface: `http://192.168.100.96`

#### FXS Port Settings (if you have an analog phone):
1. Go to **FXS PORT** tab
2. Change these settings:
   - ✅ **Account Active**: `Yes`
   - ✅ **NAT Traversal**: Change to `Keep-Alive`
   - ✅ **SIP Registration**: `Yes`
   - ✅ **Primary SIP Server**: `192.168.100.39` (already correct)
   - ✅ **SIP User ID**: `2001` (already correct)
   - ✅ **Authenticate Password**: `test123` (or any password)

3. Click **Apply** at the bottom

#### FXO Port Settings (if you have a phone line):
1. Go to **FXO PORT** tab
2. Change these settings:
   - ✅ **Account Active**: `Yes` (already correct)
   - ✅ **NAT Traversal**: Change to `Keep-Alive`
   - ✅ **SIP Registration**: `Yes`
   - ✅ **Primary SIP Server**: `192.168.100.39` (already correct)

3. Click **Apply** at the bottom

#### Reboot the Device
1. Go to **MAINTENANCE** → **Reboot**
2. Click **Reboot**
3. Wait 1-2 minutes

### Step 3: Verify Registration

Check the SIP server terminal. You should see:
```
✅ User '2001' registered from 192.168.100.96:XXXX
```

Or check the HT813 status page:
```
Port Status:
Port    Hook        User ID    Registration
FXS     On Hook     2001       Registered ✅
FXO     Idle        2001       Registered ✅
```

### Step 4: Capture Audio

Open a **NEW terminal** (keep SIP server running):

```bash
cd /home/lumi/beautyai/pabx
sudo venv/bin/python3 ht813_audio_capture.py -d 120
```

### Step 5: Make a Test Call

#### If using FXS port (analog phone):
1. Pick up the phone connected to FXS
2. Dial any number (e.g., `555`)
3. The call will be "answered" by the test server
4. **You should see RTP packets being captured!**

#### If using FXO port (PSTN line):
1. Call the phone number connected to FXO
2. The HT813 will answer and forward to SIP
3. **You should see RTP packets being captured!**

### Step 6: Convert to WAV

After capturing:

```bash
cd /home/lumi/beautyai/pabx
python3 convert_rtp_to_wav.py captures/ --all
aplay captures/session_*/audio.wav
```

## 📋 Quick Checklist

- [ ] SIP server is running (`sudo python3 simple_sip_server.py`)
- [ ] FXS/FXO port is set to "Account Active: Yes"
- [ ] NAT Traversal is set to "Keep-Alive"
- [ ] SIP Registration is "Yes"
- [ ] Primary SIP Server is "192.168.100.39"
- [ ] Device has been rebooted
- [ ] Ports show "Registered" in status page
- [ ] Capture script is running
- [ ] A call is being made through the device

## 🔧 Recommended HT813 Settings

### Critical Settings to Change:

1. **NAT Traversal**: `Keep-Alive` (not `No`)
   - This sends periodic keepalive packets to maintain connection

2. **Enable SIP OPTIONS Keep Alive**: `Yes`
   - Helps maintain registration

3. **Preferred Vocoder Choice 1**: `PCMU`
   - Best quality for testing, uncompressed

4. **Local RTP Port**: 
   - FXS: `5004` (default)
   - FXO: `5012` (default)

## 🐛 Troubleshooting

### Registration Failed

**Check SIP server terminal** - You should see REGISTER requests.

If not:
- Verify firewall allows UDP port 5060:
  ```bash
  sudo ufw allow 5060/udp
  ```
- Check HT813 can reach the server:
  ```bash
  # From HT813, it should be able to reach 192.168.100.39:5060
  ```

### No RTP Packets Captured

1. **Verify call is active** - Check SIP server shows "Call established"
2. **Check RTP ports** - Should be in range 10000-20000
3. **Run diagnostic**:
   ```bash
   sudo python3 diagnose_network.py
   ```

### Audio File is Empty

- Check codec settings - Use PCMU (G.711 μ-law)
- Verify packet count in metadata.json
- Check for packet loss

## 🎯 Alternative: Install Asterisk (Professional SIP Server)

If you want a real PBX system:

```bash
# Install Asterisk
sudo apt update
sudo apt install asterisk

# Configure extensions
sudo nano /etc/asterisk/extensions.conf

# Add:
[default]
exten => 2001,1,Answer()
exten => 2001,2,Playback(hello-world)
exten => 2001,3,Hangup()

# Configure SIP
sudo nano /etc/asterisk/pjsip.conf

# Add:
[2001]
type=endpoint
context=default
disallow=all
allow=ulaw
aor=2001

[2001]
type=aor
max_contacts=1

[2001]
type=auth
auth_type=userpass
username=2001
password=test123

# Restart Asterisk
sudo systemctl restart asterisk
```

## 📝 Summary

The HT813 needs:
1. ✅ A SIP server to register to (use simple_sip_server.py or Asterisk)
2. ✅ Active account configuration
3. ✅ NAT traversal enabled
4. ✅ An active call to generate RTP traffic

Without these, there's nothing to capture!
