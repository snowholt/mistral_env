#!/usr/bin/env python3
"""
Real-time Model Loading Monitor

This script continuously monitors the API and tracks model loading/unloading events
in real-time. It helps identify exactly when models are loaded and unloaded across
different services.

Author: BeautyAI Framework  
Date: 2025-07-27
"""

import asyncio
import aiohttp
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Set, List
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/model_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelLoadingMonitor:
    """Real-time model loading monitor."""
    
    def __init__(self, base_url: str = "http://localhost:8000", poll_interval: float = 2.0):
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.session = None
        self.previous_state = {}
        self.events = []
        self.monitoring = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_current_model_state(self) -> Dict[str, Any]:
        """Get current model loading state."""
        try:
            async with self.session.get(f"{self.base_url}/models/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"⚠️ Failed to get model status: {response.status}")
                    return {}
        except Exception as e:
            logger.warning(f"⚠️ Error getting model state: {e}")
            return {}
    
    async def get_websocket_status(self) -> Dict[str, Any]:
        """Get WebSocket service status."""
        try:
            # Try to get both WebSocket service statuses
            statuses = {}
            
            # Simple voice WebSocket status
            try:
                async with self.session.get(f"{self.base_url}/ws/simple-voice-chat/status") as response:
                    if response.status == 200:
                        statuses["simple_voice"] = await response.json()
            except Exception as e:
                logger.debug(f"Simple voice status unavailable: {e}")
            
            # Advanced voice WebSocket status
            try:
                async with self.session.get(f"{self.base_url}/ws/voice-conversation/status") as response:
                    if response.status == 200:
                        statuses["advanced_voice"] = await response.json()
            except Exception as e:
                logger.debug(f"Advanced voice status unavailable: {e}")
            
            return statuses
        except Exception as e:
            logger.debug(f"Error getting WebSocket status: {e}")
            return {}
    
    def detect_changes(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect changes in model loading state."""
        changes = []
        
        if not self.previous_state:
            # First run - record initial state
            if current_state.get("models"):
                changes.append({
                    "timestamp": time.time(),
                    "event_type": "initial_state",
                    "details": f"Found {len(current_state['models'])} models loaded at startup",
                    "models": current_state["models"]
                })
        else:
            # Compare with previous state
            prev_models = set(self.previous_state.get("models", []))
            current_models = set(current_state.get("models", []))
            
            # Models that were loaded
            new_models = current_models - prev_models
            if new_models:
                changes.append({
                    "timestamp": time.time(),
                    "event_type": "models_loaded",
                    "details": f"Models loaded: {', '.join(new_models)}",
                    "models": list(new_models)
                })
            
            # Models that were unloaded
            unloaded_models = prev_models - current_models
            if unloaded_models:
                changes.append({
                    "timestamp": time.time(),
                    "event_type": "models_unloaded",
                    "details": f"Models unloaded: {', '.join(unloaded_models)}",
                    "models": list(unloaded_models)
                })
            
            # Check for changes in model count
            prev_count = self.previous_state.get("total_loaded", 0)
            current_count = current_state.get("total_loaded", 0)
            
            if current_count != prev_count:
                changes.append({
                    "timestamp": time.time(),
                    "event_type": "model_count_changed",
                    "details": f"Model count changed: {prev_count} → {current_count}",
                    "previous_count": prev_count,
                    "current_count": current_count
                })
        
        return changes
    
    def log_event(self, event: Dict[str, Any]):
        """Log a model loading event."""
        timestamp_str = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        event_type = event["event_type"]
        details = event["details"]
        
        # Use different log levels and icons for different event types
        if event_type == "models_loaded":
            logger.info(f"🔄 [{timestamp_str}] MODEL LOADED: {details}")
        elif event_type == "models_unloaded":
            logger.info(f"📤 [{timestamp_str}] MODEL UNLOADED: {details}")
        elif event_type == "model_count_changed":
            logger.info(f"📊 [{timestamp_str}] COUNT CHANGED: {details}")
        elif event_type == "initial_state":
            logger.info(f"🚀 [{timestamp_str}] INITIAL STATE: {details}")
        else:
            logger.info(f"ℹ️ [{timestamp_str}] {event_type.upper()}: {details}")
        
        # Store event
        self.events.append(event)
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        logger.info(f"🔍 Starting model loading monitor (polling every {self.poll_interval}s)")
        logger.info(f"📊 Monitoring API at: {self.base_url}")
        logger.info(f"⚠️ Press Ctrl+C to stop monitoring")
        
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Get current model state
                current_state = await self.get_current_model_state()
                
                if current_state:
                    # Detect and log changes
                    changes = self.detect_changes(current_state)
                    for change in changes:
                        self.log_event(change)
                    
                    # Update previous state
                    self.previous_state = current_state
                    
                    # Log current status (every 30 seconds)
                    if len(self.events) % 15 == 0:  # Every 15 polls = ~30 seconds
                        total_loaded = current_state.get("total_loaded", 0)
                        models = current_state.get("models", [])
                        logger.info(f"📊 Current Status: {total_loaded} models loaded {models}")
                
                # Sleep until next poll
                await asyncio.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Monitoring stopped by user")
                self.monitoring = False
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def run_targeted_test(self, test_type: str = "websocket"):
        """Run a targeted test while monitoring."""
        logger.info(f"\n🎯 Running targeted test: {test_type}")
        
        if test_type == "websocket":
            await self.test_websocket_impact()
        elif test_type == "chat":
            await self.test_chat_impact()
        else:
            logger.warning(f"⚠️ Unknown test type: {test_type}")
    
    async def test_websocket_impact(self):
        """Test the impact of WebSocket connections on model loading."""
        logger.info("🔌 Testing WebSocket connection impact on model loading...")
        
        try:
            import websockets
            
            # Connect to simple voice WebSocket
            ws_url = self.base_url.replace("http://", "ws://") + "/ws/simple-voice-chat?language=ar&voice_type=female"
            
            logger.info(f"🔗 Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ WebSocket connected - check for model loading events above")
                
                # Keep connection alive for a few seconds to see the impact
                await asyncio.sleep(10)
                
                logger.info("🔚 Closing WebSocket connection")
                
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
    
    async def test_chat_impact(self):
        """Test the impact of chat API requests on model loading."""
        logger.info("💬 Testing Chat API impact on model loading...")
        
        try:
            chat_request = {
                "model_name": "qwen3-unsloth-q4ks",
                "message": "مرحبا، هذا اختبار سريع",
                "max_new_tokens": 20
            }
            
            logger.info("📤 Sending chat request...")
            
            async with self.session.post(
                f"{self.base_url}/inference/chat",
                json=chat_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ Chat request completed - check for model loading events above")
                    logger.info(f"   Response: {result.get('response', '')[:50]}...")
                else:
                    logger.error(f"❌ Chat request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Chat test failed: {e}")
    
    def save_events(self, output_file: str = None):
        """Save monitored events to file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/lumi/beautyai/unitTests_scripts/model_loading_analysis/monitor_events_{timestamp}.json"
        
        events_data = {
            "monitoring_session": {
                "start_time": self.events[0]["timestamp"] if self.events else time.time(),
                "end_time": time.time(),
                "total_events": len(self.events),
                "poll_interval": self.poll_interval
            },
            "events": self.events
        }
        
        with open(output_file, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"💾 Events saved to: {output_file}")
        return output_file


async def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BeautyAI Model Loading Monitor")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--test", choices=["websocket", "chat"], help="Run a specific test while monitoring")
    parser.add_argument("--duration", type=int, default=0, help="Monitor for specific duration (seconds, 0=infinite)")
    
    args = parser.parse_args()
    
    try:
        async with ModelLoadingMonitor(args.url, args.interval) as monitor:
            
            # Start monitoring task
            monitor_task = asyncio.create_task(monitor.monitor_loop())
            
            # Run targeted test if specified
            if args.test:
                # Wait a bit for initial monitoring
                await asyncio.sleep(5)
                await monitor.run_targeted_test(args.test)
            
            # Handle duration limit
            if args.duration > 0:
                await asyncio.sleep(args.duration)
                monitor.monitoring = False
                logger.info(f"⏰ Monitoring duration ({args.duration}s) reached")
            
            # Wait for monitoring to complete
            await monitor_task
            
            # Save events
            output_file = monitor.save_events()
            logger.info(f"✅ Monitoring complete. Events saved to: {output_file}")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
