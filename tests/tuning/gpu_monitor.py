#!/usr/bin/env python3
"""
GPU Monitor Script for LlamaCpp Engine
Monitors real-time GPU utilization during inference
"""

import subprocess
import time
import threading
import signal
import sys

class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 GPU monitoring started (Ctrl+C to stop)")
        
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("\n🛑 GPU monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        print(f"{'Time':<12} {'GPU%':<6} {'MEM%':<6} {'VRAM':<10} {'TEMP':<6} {'PWR':<8}")
        print("-" * 55)
        
        while self.monitoring:
            try:
                # Get GPU stats using nvidia-smi
                cmd = [
                    "nvidia-smi", 
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    stats = result.stdout.strip().split(', ')
                    if len(stats) >= 5:
                        gpu_util = stats[0].strip()
                        mem_util = stats[1].strip()
                        vram_used = stats[2].strip()
                        temp = stats[3].strip()
                        power = stats[4].strip()
                        
                        current_time = time.strftime("%H:%M:%S")
                        
                        print(f"{current_time:<12} {gpu_util}%{'':<3} {mem_util}%{'':<3} {vram_used}MB{'':<4} {temp}°C{'':<2} {power}W")
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
                print(f"Error getting GPU stats: {e}")
                
            time.sleep(self.interval)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Stopping GPU monitor...")
    monitor.stop_monitoring()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 LlamaCpp GPU Monitor")
    print("=" * 50)
    print("This script monitors GPU utilization in real-time.")
    print("Run your LlamaCpp inference in another terminal to see GPU usage.")
    print("Press Ctrl+C to stop monitoring.")
    print("=" * 50)
    
    # Create and start monitor
    monitor = GPUMonitor(interval=0.5)  # Update every 0.5 seconds
    
    try:
        monitor.start_monitoring()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        sys.exit(0)
