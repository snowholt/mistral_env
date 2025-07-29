# BeautyAI Web UI - Microphone Fix

## The Issue
Modern browsers (Chrome, Firefox, Safari) require HTTPS to access the microphone for security reasons. If you see the error "❌ Audio recording is not supported in your browser", it's likely because you're accessing the app via HTTP.

## Quick Solutions

### Solution 1: Use localhost (Immediate Fix)
Access your app via `http://localhost:5000` instead of `http://127.0.0.1:5000` or your IP address. Browsers allow microphone access on localhost without HTTPS.

### Solution 2: Enable HTTPS (Recommended)

1. **Generate SSL certificate:**
   ```bash
   cd src/web_ui
   python generate_ssl_cert.py
   ```

2. **Start the app:**
   ```bash
   python start_ui.py
   ```
   
3. **Access via HTTPS:**
   ```
   https://localhost:5000
   ```

4. **Accept the self-signed certificate warning** in your browser (this is safe for development)

## What This Fixes

✅ **Microphone access from any device**  
✅ **Works with IP addresses (e.g., `https://192.168.1.100:5000`)**  
✅ **Secure audio recording**  
✅ **Better browser compatibility**  

## Browser-Specific Notes

### Chrome
- Will show "Not secure" warning for self-signed certificates
- Click "Advanced" → "Proceed to localhost (unsafe)" to continue
- Microphone will work after accepting the certificate

### Firefox
- Will show "Warning: Potential Security Risk"
- Click "Advanced" → "Accept the Risk and Continue"
- Microphone will work after accepting the certificate

### Safari
- Similar warning for self-signed certificates
- Accept the certificate to enable microphone access

## Troubleshooting

### If microphone still doesn't work:

1. **Check browser permissions:**
   - Chrome: `chrome://settings/content/microphone`
   - Firefox: `about:preferences#privacy`
   - Ensure the site isn't blocked

2. **Clear browser cache and cookies**

3. **Try incognito/private mode**

4. **Check if another app is using the microphone**

5. **Test microphone at:** https://webcammictest.com/check-mic

### Error Messages and Solutions:

| Error | Solution |
|-------|----------|
| "Audio recording is not supported" | Use HTTPS or localhost |
| "Microphone access denied" | Check browser permissions |
| "No microphone found" | Check hardware connection |
| "Microphone is being used by another application" | Close other apps using microphone |

## Development Notes

The enhanced JavaScript code now includes:
- ✅ HTTPS/localhost security checks
- ✅ Detailed error messages
- ✅ Permission status checking
- ✅ Better error handling for different failure scenarios

## Files Modified

- `app.py` - Added HTTPS support
- `generate_ssl_cert.py` - SSL certificate generator
- `start_ui.py` - Enhanced startup messages
- `static/js/main.js` - Improved audio permission handling

Enjoy using BeautyAI with full microphone support! 🎤✨
