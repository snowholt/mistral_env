#!/usr/bin/env python3
"""
Enhanced Service Monitoring Tool for BeautyAI

This tool provides comprehensive monitoring and analysis of the BeautyAI API service.
It can capture logs, analyze them, and provide insights about service health.
"""

import subprocess
import argparse
import datetime
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ServiceAnalyzer:
    def __init__(self, service_name="beautyai-api.service", log_dir="logs/service"):
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_logs(self, log_content: str) -> Dict:
        """Analyze service logs and extract insights"""
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "service": self.service_name,
            "status": "unknown",
            "startup_time": None,
            "errors": [],
            "warnings": [],
            "info_messages": [],
            "memory_usage": None,
            "cpu_usage": None,
            "issues": [],
            "recommendations": [],
            "webrtc_stats": {
                "connections_total": 0,
                "connections_active": 0,
                "connections_failed": 0,
                "ice_negotiations": [],
                "sdp_exchanges": [],
                "audio_quality_issues": []
            }
        }
        
        lines = log_content.split('\n')
        
        for line in lines:
            # Extract service status
            if "Active: active (running)" in line:
                analysis["status"] = "running"
            elif "Active: failed" in line:
                analysis["status"] = "failed"
            elif "Active: inactive" in line:
                analysis["status"] = "stopped"
                
            # Extract startup time
            startup_match = re.search(r'since (.+?);', line)
            if startup_match:
                analysis["startup_time"] = startup_match.group(1)
                
            # Extract memory usage
            memory_match = re.search(r'Memory: (.+?) \(max:', line)
            if memory_match:
                analysis["memory_usage"] = memory_match.group(1)
                
            # Extract CPU usage
            cpu_match = re.search(r'CPU: (.+?)s', line)
            if cpu_match:
                analysis["cpu_usage"] = cpu_match.group(1)
                
            # Extract JSON log messages
            if '{"timestamp":' in line:
                try:
                    json_part = line.split('{"timestamp":', 1)[1]
                    json_part = '{"timestamp":' + json_part
                    log_data = json.loads(json_part)
                    
                    if log_data.get("levelname") == "ERROR":
                        analysis["errors"].append({
                            "timestamp": log_data.get("timestamp"),
                            "message": log_data.get("message")
                        })
                    elif log_data.get("levelname") == "WARNING":
                        analysis["warnings"].append({
                            "timestamp": log_data.get("timestamp"),
                            "message": log_data.get("message")
                        })
                    elif log_data.get("levelname") == "INFO":
                        analysis["info_messages"].append({
                            "timestamp": log_data.get("timestamp"),
                            "message": log_data.get("message")
                        })
                except json.JSONDecodeError:
                    pass
                    
            # Detect common issues
            if "timeout" in line.lower():
                analysis["issues"].append("Service timeout detected")
            if "killed" in line.lower() and "sigkill" in line.lower():
                analysis["issues"].append("Service was forcefully killed")
            if "failed to start" in line.lower():
                analysis["issues"].append("Service failed to start")
            if "out of memory" in line.lower() or "oom" in line.lower():
                analysis["issues"].append("Out of memory condition")
            if "config system not available" in line.lower():
                analysis["issues"].append("Configuration system unavailable")
            
            # Detect WebRTC-specific patterns
            if "webrtc" in line.lower():
                # Connection tracking
                if "connection established" in line.lower():
                    analysis["webrtc_stats"]["connections_total"] += 1
                    analysis["webrtc_stats"]["connections_active"] += 1
                elif "connection failed" in line.lower():
                    analysis["webrtc_stats"]["connections_failed"] += 1
                elif "connection closed" in line.lower() or "disconnected" in line.lower():
                    if analysis["webrtc_stats"]["connections_active"] > 0:
                        analysis["webrtc_stats"]["connections_active"] -= 1
                
                # ICE negotiation tracking
                if "ice" in line.lower():
                    if "negotiation" in line.lower() or "candidate" in line.lower():
                        analysis["webrtc_stats"]["ice_negotiations"].append({
                            "timestamp": line.split()[0:2],
                            "status": "completed" if "completed" in line.lower() else "in_progress"
                        })
                    if "failed" in line.lower():
                        analysis["issues"].append("WebRTC ICE negotiation failure detected")
                
                # SDP exchange tracking
                if "sdp" in line.lower():
                    if "offer" in line.lower():
                        analysis["webrtc_stats"]["sdp_exchanges"].append({
                            "type": "offer",
                            "timestamp": line.split()[0:2]
                        })
                    elif "answer" in line.lower():
                        analysis["webrtc_stats"]["sdp_exchanges"].append({
                            "type": "answer",
                            "timestamp": line.split()[0:2]
                        })
                
                # Audio quality issues
                if "packet loss" in line.lower() or "jitter" in line.lower():
                    analysis["webrtc_stats"]["audio_quality_issues"].append({
                        "issue": "high_packet_loss" if "packet loss" in line.lower() else "high_jitter",
                        "timestamp": line.split()[0:2]
                    })
                
                # VAD-related issues
                if "vad" in line.lower() and ("false positive" in line.lower() or "failed" in line.lower()):
                    analysis["issues"].append("WebRTC VAD detection issues")
                
        # Generate recommendations
        if analysis["issues"]:
            if "Service timeout detected" in analysis["issues"]:
                analysis["recommendations"].append("Consider increasing TimeoutStopSec in service file")
            if "Service was forcefully killed" in analysis["issues"]:
                analysis["recommendations"].append("Service may not be shutting down gracefully - check application shutdown handlers")
            if "Configuration system unavailable" in analysis["issues"]:
                analysis["recommendations"].append("Check WebSocket pool configuration and dependencies")
            if "WebRTC ICE negotiation failure detected" in analysis["issues"]:
                analysis["recommendations"].append("Check STUN/TURN server configuration and network connectivity")
            if "WebRTC VAD detection issues" in analysis["issues"]:
                analysis["recommendations"].append("Review VAD thresholds and consider language-specific tuning")
        
        # WebRTC-specific recommendations
        if analysis["webrtc_stats"]["connections_failed"] > 0:
            failure_rate = analysis["webrtc_stats"]["connections_failed"] / max(analysis["webrtc_stats"]["connections_total"], 1)
            if failure_rate > 0.1:  # > 10% failure rate
                analysis["recommendations"].append(f"High WebRTC connection failure rate ({failure_rate:.1%}) - investigate network/signaling issues")
        
        if len(analysis["webrtc_stats"]["audio_quality_issues"]) > 3:
            analysis["recommendations"].append("Multiple audio quality issues detected - check network bandwidth and latency")
                
        return analysis
        
    def capture_and_analyze(self, lines=200) -> Tuple[str, Dict]:
        """Capture logs and perform analysis"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"service_logs_{timestamp}.log"
        analysis_file = self.log_dir / f"service_analysis_{timestamp}.json"
        
        cmd = [
            "sudo", "journalctl", 
            "-u", self.service_name,
            "-n", str(lines),
            "--no-pager"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Save raw logs
            with open(log_file, 'w') as f:
                f.write(f"# Service logs captured at {datetime.datetime.now()}\n")
                f.write(f"# Service: {self.service_name}\n")
                f.write(f"# Lines: {lines}\n")
                f.write("# " + "="*60 + "\n\n")
                f.write(result.stdout)
                
            # Perform analysis
            analysis = self.analyze_logs(result.stdout)
            
            # Save analysis
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            return str(log_file), analysis
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error capturing logs: {e}")
            return None, None
            
    def print_analysis_summary(self, analysis: Dict):
        """Print a human-readable analysis summary"""
        print("\n" + "="*60)
        print("🔍 SERVICE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Service: {analysis['service']}")
        print(f"⏰ Analysis Time: {analysis['timestamp']}")
        print(f"🟢 Status: {analysis['status'].upper()}")
        
        if analysis['startup_time']:
            print(f"🚀 Started: {analysis['startup_time']}")
            
        if analysis['memory_usage']:
            print(f"🧠 Memory: {analysis['memory_usage']}")
            
        if analysis['cpu_usage']:
            print(f"⚡ CPU: {analysis['cpu_usage']}s")
            
        if analysis['errors']:
            print(f"\n❌ ERRORS ({len(analysis['errors'])}):")
            for error in analysis['errors'][-3:]:  # Show last 3 errors
                print(f"   • {error['message']}")
                
        if analysis['warnings']:
            print(f"\n⚠️  WARNINGS ({len(analysis['warnings'])}):")
            for warning in analysis['warnings'][-3:]:  # Show last 3 warnings
                print(f"   • {warning['message']}")
                
        if analysis['issues']:
            print(f"\n🚨 ISSUES DETECTED:")
            for issue in analysis['issues']:
                print(f"   • {issue}")
                
        if analysis['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
        
        # WebRTC Statistics
        if analysis.get('webrtc_stats'):
            webrtc = analysis['webrtc_stats']
            if (webrtc['connections_total'] > 0 or 
                len(webrtc['ice_negotiations']) > 0 or 
                len(webrtc['sdp_exchanges']) > 0):
                print(f"\n🌐 WEBRTC STATISTICS:")
                print(f"   Connections Total: {webrtc['connections_total']}")
                print(f"   Connections Active: {webrtc['connections_active']}")
                print(f"   Connections Failed: {webrtc['connections_failed']}")
                if webrtc['connections_total'] > 0:
                    success_rate = ((webrtc['connections_total'] - webrtc['connections_failed']) / 
                                  webrtc['connections_total'] * 100)
                    print(f"   Connection Success Rate: {success_rate:.1f}%")
                print(f"   ICE Negotiations: {len(webrtc['ice_negotiations'])}")
                print(f"   SDP Exchanges: {len(webrtc['sdp_exchanges'])}")
                if webrtc['audio_quality_issues']:
                    print(f"   Audio Quality Issues: {len(webrtc['audio_quality_issues'])}")
                
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Enhanced BeautyAI service monitoring")
    parser.add_argument("--service", default="beautyai-api.service", 
                        help="Service name (default: beautyai-api.service)")
    parser.add_argument("--lines", type=int, default=200, 
                        help="Number of lines to capture (default: 200)")
    parser.add_argument("--analyze", action="store_true", 
                        help="Perform analysis on captured logs")
    parser.add_argument("--output-dir", default="logs/service", 
                        help="Output directory for logs")
    parser.add_argument("--summary", action="store_true", 
                        help="Show analysis summary only")
    
    args = parser.parse_args()
    
    analyzer = ServiceAnalyzer(args.service, args.output_dir)
    
    if args.analyze or args.summary:
        log_file, analysis = analyzer.capture_and_analyze(args.lines)
        
        if analysis:
            if not args.summary:
                print(f"✅ Logs captured to: {log_file}")
                print(f"📋 Analysis saved to: {log_file.replace('.log', '.json')}")
                
            analyzer.print_analysis_summary(analysis)
        else:
            print("❌ Failed to capture and analyze logs")
    else:
        print("Use --analyze or --summary to capture and analyze logs")
        print("Example: python3 tools/service_analyzer.py --analyze")

if __name__ == "__main__":
    main()