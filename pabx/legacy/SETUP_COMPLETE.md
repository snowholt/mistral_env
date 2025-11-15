# HT813 Setup Complete! 🎉

## What We've Accomplished

✅ **Network Configuration**
- Fixed VPN routing to allow access to HT813 from remote locations
- Added NAT/MASQUERADE rules for VPN ↔ LAN traffic
- Configured UFW firewall rules for proper routing
- Confirmed HT813 is accessible at `http://192.168.100.96`

✅ **Audio Capture Tools**
- Created `ht813_audio_capture.py` - Professional RTP packet capture tool
- Created `convert_rtp_to_wav.py` - Convert captured audio to playable WAV format
- Installed all required Python dependencies in `pabx/venv`
- Supports multiple codecs: PCMU, PCMA, G.722, etc.

✅ **Testing & Helper Scripts**
- `test_connectivity.sh` - Verify HT813 connectivity and configuration
- `demo_capture.sh` - Quick demo for capturing audio
- `setup_vpn_nat.sh` - VPN NAT setup script for future reference
- Comprehensive `README.md` with detailed instructions

## Quick Reference

### Access HT813 Web Interface
```bash
# From local network or VPN
http://192.168.100.96

# Default credentials
Username: admin
Password: admin
```

### Capture Audio (Basic)
```bash
cd /home/lumi/beautyai/pabx

# Capture for 60 seconds
sudo venv/bin/python3 ht813_audio_capture.py -d 60

# Convert to WAV
venv/bin/python3 convert_rtp_to_wav.py captures/ --all

# Play audio
aplay captures/session_*/audio.wav
```

### Run Quick Demo
```bash
cd /home/lumi/beautyai/pabx
./demo_capture.sh
```

### Test Connectivity
```bash
cd /home/lumi/beautyai/pabx
./test_connectivity.sh
```

## Device Information

**Grandstream HT813 Analog Telephone Adapter**
- **IP Address**: 192.168.100.96
- **MAC Address**: EC:74:D7:62:4E:34 (LAN)
- **Model**: HT813 (Hardware V1.1E)
- **Firmware**: 1.0.17.3
- **Uptime**: 2+ days

**Ports:**
- **FXS Port**: Connect analog phones (User ID: 2001, Not Registered)
- **FXO Port**: Connect PSTN lines (User ID: 2001, Registered ✓)

## How It Works

1. **RTP Capture**: The script captures UDP packets in the RTP port range (10000-20000)
2. **Packet Parsing**: Parses RTP headers to extract audio payload
3. **Session Tracking**: Groups packets by SSRC (synchronization source)
4. **Audio Extraction**: Saves raw audio data and metadata
5. **Codec Conversion**: Converts from G.711/G.722 to PCM WAV format

## Configuration for Your Friend

To make calls through the HT813, your friend needs to:

### Option 1: Use FXO Port (Existing Setup)
The FXO port is already registered. Just connect a phone line and it will work.

### Option 2: Configure FXS Port for SIP
1. Access `http://192.168.100.96`
2. Go to **FXS Port** → **Account 1**
3. Configure:
   - Primary SIP Server: `192.168.100.39` (or your SIP server)
   - SIP User ID: `2001`
   - Preferred Codec: `PCMU (G.711μ)`
   - RTP Port Range: `10000-20000`
4. Connect an analog phone to the FXS port
5. Start capture and make a test call

## What You Need to Do

### To Capture Audio:

1. **Tell your friend to make a call** through the HT813
2. **Run the capture script** before the call:
   ```bash
   sudo venv/bin/python3 ht813_audio_capture.py -d 60
   ```
3. **Wait for the call to complete**
4. **Convert the captured audio**:
   ```bash
   venv/bin/python3 convert_rtp_to_wav.py captures/ --all
   ```
5. **Listen to the result**:
   ```bash
   aplay captures/session_*/audio.wav
   ```

### Configuration Needed (If Using FXS Port):

The FXS port needs to be registered to a SIP server. You have two options:

**A. Use an existing SIP provider**
- Configure the HT813 to register to services like VoIP.ms, Twilio, etc.

**B. Set up your own SIP server (Asterisk)**
- This gives you full control but requires more setup
- I can help you set this up if needed!

## Next Steps

1. **Test the capture** by having your friend make a call
2. **Verify audio quality** by listening to the captured WAV files
3. **Integrate with BeautyAI** if you want to process the audio through your AI models
4. **Set up a SIP server** if you want more control over call routing

## Troubleshooting

### No Audio Captured
- Verify the HT813 is making/receiving calls during capture
- Check that RTP port range matches (10000-20000)
- Ensure firewall allows UDP traffic on those ports

### Can't Access from VPN
- Reconnect your VPN connection
- Verify UFW rules: `sudo ufw status | grep 192.168.100`
- Check NAT rule: `sudo iptables -t nat -L POSTROUTING -n`

### Poor Audio Quality
- Use PCMU codec (best quality, uncompressed)
- Check for packet loss in metadata.json
- Verify network stability

## Files Created

```
pabx/
├── README.md                      # Comprehensive documentation
├── SETUP_COMPLETE.md             # This file
├── requirements.txt              # Python dependencies
├── venv/                         # Virtual environment
├── ht813_audio_capture.py       # Main capture script
├── convert_rtp_to_wav.py        # Audio conversion tool
├── test_connectivity.sh         # Connectivity test
├── demo_capture.sh              # Quick demo
├── setup_vpn_nat.sh            # VPN setup helper
├── HT813_User_Guide.pdf        # Official documentation
└── captures/                    # Output directory (created on first run)
```

## Support

If you need help:
1. Check the `README.md` for detailed instructions
2. Run `./test_connectivity.sh` to diagnose issues
3. Review the HT813 User Guide PDF
4. Ask me! 💜

---

**Ready to capture some audio!** 🎤✨

Just run `./demo_capture.sh` when someone is making a call through the HT813!
