# React Frontend - Quick Start Guide

## 🎉 Phase 5 Complete!

The React frontend is now ready with all components built!

## What's Included

### Components ✅
- **CallList** - Real-time active calls display
- **CallDetails** - Call statistics and control buttons
- **DeviceDashboard** - HT813 status and FXS port monitoring
- **CaptureStatus** - Packet capture statistics

### Features ✅
- **Real-time Updates** via WebSocket
- **API Integration** with axios
- **State Management** with Zustand
- **TypeScript** for type safety
- **Modern UI** with Lucide icons
- **Dark Mode** support

## Installation & Setup

### 1. Install Dependencies

```bash
cd /home/lumi/beautyai/pabx/ui
npm install
```

This will install:
- React 18 + React DOM
- TypeScript
- Vite (build tool)
- Axios (HTTP client)
- Zustand (state management)
- Lucide React (icons)
- date-fns (date formatting)

### 2. Start Development Server

```bash
# Make sure backend is running first!
cd /home/lumi/beautyai/pabx
./run_server.py --mode api

# In another terminal, start frontend
cd /home/lumi/beautyai/pabx/ui
npm run dev
```

The UI will be available at: **http://localhost:3000**

## How It Works

### API Proxy
Vite dev server automatically proxies API requests:
- `/api/*` → `http://localhost:8080/api/*`
- `/ws` → `ws://localhost:8080/ws`

So you can just call `/api/calls` and it works!

### WebSocket Connection
The `useWebSocket` hook automatically connects to the backend and provides real-time call events:

```typescript
const { isConnected, lastEvent } = useWebSocket();

// lastEvent contains:
// - type: 'call_incoming' | 'call_answered' | 'call_ended'
// - data: Call object
```

### State Management
Zustand store manages global state:

```typescript
const { calls, setCalls, addCall, updateCall } = useStore();
```

## File Structure

```
ui/
├── src/
│   ├── components/
│   │   ├── CallList.tsx         # Active calls (polls every 2s)
│   │   ├── CallDetails.tsx      # Call stats (polls every 2s)
│   │   ├── DeviceDashboard.tsx  # HT813 (polls every 5s)
│   │   └── CaptureStatus.tsx    # Capture (polls every 3s)
│   ├── hooks/
│   │   └── useWebSocket.ts      # WebSocket hook with auto-reconnect
│   ├── services/
│   │   └── api.ts               # API service with TypeScript types
│   ├── store/
│   │   └── useStore.ts          # Zustand state management
│   ├── App.tsx                  # Main app with 2x2 grid layout
│   ├── main.tsx                 # React entry point
│   ├── index.css                # Global styles + dark mode
│   └── vite-env.d.ts           # Vite types
├── index.html                   # HTML template
├── package.json                 # Dependencies
├── tsconfig.json                # TypeScript config
├── vite.config.ts               # Vite config with proxy
└── README.md                    # UI documentation
```

## Production Build

```bash
cd /home/lumi/beautyai/pabx/ui
npm run build
```

This creates an optimized production build in `dist/` directory.

### Serve with FastAPI

You can serve the built files directly from FastAPI:

```python
from fastapi.staticfiles import StaticFiles

# After all API routes
app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")
```

Then visit: **http://localhost:8080** (no port 3000 needed!)

## Usage Examples

### View Active Calls
1. Open http://localhost:3000
2. Make a call from HT813
3. See it appear in real-time in "Active Calls"
4. Click on call to see details

### Monitor HT813
- View IP, MAC, firmware version
- Check FXS port registration status
- See call statistics per port
- Reboot device remotely

### Control Calls
- Click "End Call" to terminate
- Click "Record" to start recording
- View RTP statistics (packets, bytes, jitter, loss)

### Packet Capture
- Shows if capture is enabled/running
- Displays packet count and capture rate
- Shows packet type breakdown (SIP/RTP/RTCP)

## Troubleshooting

### Can't Connect to Backend

**Problem**: UI shows "Disconnected" or API errors

**Solution**:
```bash
# 1. Check backend is running
curl http://localhost:8080/api/health

# 2. If not, start it
cd /home/lumi/beautyai/pabx
./run_server.py --mode api

# 3. Restart UI
cd ui
npm run dev
```

### WebSocket Won't Connect

**Problem**: Connection indicator stays red

**Solution**:
1. Check backend WebSocket: `wscat -c ws://localhost:8080/ws`
2. Check browser console for errors
3. Verify firewall allows WebSocket connections

### Build Fails

**Problem**: `npm run build` fails

**Solution**:
```bash
# Clear everything and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

### TypeScript Errors

**Problem**: Type errors during development

**Solution**:
- Check `src/services/api.ts` for correct type definitions
- Ensure backend API matches expected response format
- Run `npm run build` to see all type errors

## Development Tips

### Auto-reload
- Vite has hot module replacement (HMR)
- Changes to `.tsx` files reload instantly
- Changes to `.ts` files may require manual refresh

### API Testing
```bash
# Test backend directly
curl http://localhost:8080/api/calls
curl http://localhost:8080/api/ht813/status
curl http://localhost:8080/api/capture/status
```

### State Debugging
Add to any component:
```typescript
import { useStore } from '../store/useStore';

const state = useStore();
console.log('Current state:', state);
```

### WebSocket Debugging
Open browser DevTools → Network → WS → Click the WebSocket connection to see messages

## Next Steps

1. **Customize Styling**: Edit `src/index.css` for colors/fonts
2. **Add Features**: Create new components in `src/components/`
3. **Production Deploy**: Build and serve from FastAPI or static hosting
4. **Mobile Support**: Add responsive breakpoints for mobile devices

## Summary

✅ **React 18** + TypeScript + Vite  
✅ **Real-time updates** via WebSocket  
✅ **4 main components** (Calls, Details, Device, Capture)  
✅ **API integration** with automatic polling  
✅ **State management** with Zustand  
✅ **Modern UI** with dark mode  
✅ **Production ready** with build system  

**Total files created: 16**  
**Ready to use!** 🎉
