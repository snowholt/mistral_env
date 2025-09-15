#!/usr/bin/env python3
"""
Service Log Capture Tool for BeautyAI

This tool captures systemd journal logs for the BeautyAI API service
and saves them to files for analysis. It provides both one-time capture
and continuous monitoring options.
"""

import subprocess
import argparse
import datetime
import os
import signal
import sys
from pathlib import Path

class ServiceLogCapture:
    def __init__(self, service_name="beautyai-api.service", log_dir="logs/service"):
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        
    def capture_current_logs(self, lines=200):
        """Capture current logs without following"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.log_dir / f"service_logs_{timestamp}.log"
        
        cmd = [
            "sudo", "journalctl", 
            "-u", self.service_name,
            "-n", str(lines),
            "--no-pager"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            with open(output_file, 'w') as f:
                f.write(f"# Service logs captured at {datetime.datetime.now()}\n")
                f.write(f"# Service: {self.service_name}\n")
                f.write(f"# Lines: {lines}\n")
                f.write("# " + "="*60 + "\n\n")
                f.write(result.stdout)
                
            print(f"✅ Logs captured to: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error capturing logs: {e}")
            return None
            
    def follow_logs(self, output_file=None):
        """Follow logs in real-time and save to file"""
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.log_dir / f"service_follow_{timestamp}.log"
            
        cmd = [
            "sudo", "journalctl",
            "-u", self.service_name,
            "-f", "-n", "50"
        ]
        
        print(f"📝 Following logs for {self.service_name}...")
        print(f"💾 Saving to: {output_file}")
        print("⛔ Press Ctrl+C to stop")
        
        try:
            with open(output_file, 'w') as f:
                f.write(f"# Service logs follow started at {datetime.datetime.now()}\n")
                f.write(f"# Service: {self.service_name}\n")
                f.write("# " + "="*60 + "\n\n")
                f.flush()
                
                self.process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in self.process.stdout:
                    print(line.rstrip())  # Display to console
                    f.write(line)         # Save to file
                    f.flush()             # Ensure immediate write
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping log capture...")
            if self.process:
                self.process.terminate()
            print(f"✅ Logs saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error following logs: {e}")
            
    def get_service_status(self):
        """Get current service status"""
        cmd = ["sudo", "systemctl", "status", self.service_name, "--no-pager", "-l"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            return f"Error getting status: {e}"

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Interrupted by user')
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Capture BeautyAI service logs")
    parser.add_argument("--service", default="beautyai-api.service", 
                        help="Service name (default: beautyai-api.service)")
    parser.add_argument("--follow", action="store_true", 
                        help="Follow logs in real-time")
    parser.add_argument("--lines", type=int, default=200, 
                        help="Number of lines to capture (default: 200)")
    parser.add_argument("--status", action="store_true", 
                        help="Show service status")
    parser.add_argument("--output-dir", default="logs/service", 
                        help="Output directory for logs")
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    capture = ServiceLogCapture(args.service, args.output_dir)
    
    if args.status:
        print("📊 Service Status:")
        print("=" * 50)
        status = capture.get_service_status()
        print(status)
        print("=" * 50)
        
    if args.follow:
        capture.follow_logs()
    else:
        capture.capture_current_logs(args.lines)

if __name__ == "__main__":
    main()