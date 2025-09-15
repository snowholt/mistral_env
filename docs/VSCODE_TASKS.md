# VS Code Tasks Reference - BeautyAI

## Enhanced Tasks with Icons 🎯

### 🚀 Service Management - API
- **🚀 Service: API - Start** - Start the BeautyAI API service
- **🛑 Service: API - Stop** - Stop the BeautyAI API service  
- **📊 Service: API - Status** - Show detailed service status
- **📝 Service: API - Journal (Follow)** - Follow service logs (5min timeout)

### 🌐 Service Management - WebUI
- **🚀 Service: WebUI - Start** - Start the WebUI service
- **🛑 Service: WebUI - Stop** - Stop the WebUI service
- **📊 Service: WebUI - Status** - Show WebUI service status

### 🔧 Development Mode
- **🔥 Dev: Run API (direct uvicorn script)** - Run API in development mode (default build task)
- **🔪 Dev: Kill Direct API Processes** - Kill running development API processes
- **📋 Dev: Tail Uvicorn Live (Direct Mode)** - Follow development logs (5min timeout)
- **🧹 Dev: Clear Uvicorn Log File** - Clear the development log file

### 🧪 Testing
- **🧪 Test: Streaming - Single PCM (Q1 Arabic)** - Test single PCM file streaming
- **🧪 Test: Streaming - Pytest Suite** - Run full streaming test suite

### 🔍 Enhanced Monitoring & Analysis ⭐ **NEW**
- **🔍 Monitor: Service Analysis (Quick)** - Quick service health check
- **🔍 Monitor: Service Analysis (Detailed)** - Detailed analysis with 500 log lines
- **📋 Monitor: Capture Service Logs** - Capture logs with status
- **📺 Monitor: Follow Logs (5min timeout)** - Follow logs with auto-timeout

### 🛠️ Utilities ⭐ **NEW**
- **🧹 Utility: Clear All Logs** - Clear all log files
- **📊 Utility: Show Log Files** - List all available log files
- **🔄 Utility: Restart API Service** - Stop and start API service

## Key Improvements ✅

### 1. **Fixed Background Task Issues**
- Removed problematic `isBackground: true` flags
- Added timeout protection (5 minutes) for long-running tasks
- Better error handling with fallback messages

### 2. **Enhanced Visual Experience**
- Unicode icons for easy task identification
- Improved presentation settings (silent/clear/focus)
- Better panel management (shared/dedicated)

### 3. **Added Problem Matchers**
- Python error detection for development tasks
- Pytest pattern matching for test failures
- Errors now appear in VS Code Problems panel

### 4. **New Monitoring Capabilities**
- Integration with custom service analyzer tools
- Quick health checks without saving files
- Detailed analysis with JSON output
- Smart log capture and following

## Quick Access Tips 💡

### Keyboard Shortcuts
1. **Ctrl+Shift+P** → "Tasks: Run Task"
2. Type task name or use arrow keys to select
3. Icons help identify task categories quickly

### Recommended Workflow
1. **🔍 Monitor: Service Analysis (Quick)** - Check service health
2. **🔥 Dev: Run API** - Start development if needed
3. **📺 Monitor: Follow Logs** - Watch real-time activity
4. **🧪 Test: Streaming** - Run tests when ready

### When Tasks Get "Cancelled"
✅ **FIXED!** - Tasks now have proper timeouts and won't hang indefinitely
- Journal/log following automatically stops after 5 minutes
- Clear error messages when timeouts occur
- No more indefinite background processes

## Files Created/Modified

### Enhanced Tools
- `tools/service_analyzer.py` - Intelligent service analysis
- `tools/capture_service_logs.py` - Professional log capture
- `.vscode/tasks.json` - Updated with icons and improvements

### Log Storage
- `logs/service/` - Service analysis and captured logs
- Automatic timestamping and JSON analysis files
- Smart cleanup utilities included