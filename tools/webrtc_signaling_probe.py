#!/usr/bin/env python3
"""
WebRTC Signaling Probe Tool

Local testing tool for WebRTC signaling handshake. Tests SDP offer/answer
exchange and ICE candidate handling without full browser environment.

Usage:
    python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa
    python tools/webrtc_signaling_probe.py --url http://localhost:8000 --verbose

Author: BeautyAI Framework
Date: October 15, 2025
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

try:
    import requests
    from requests.exceptions import RequestException, ConnectionError, Timeout
except ImportError:
    print("❌ Error: 'requests' library not installed")
    print("   Install with: pip install requests")
    sys.exit(1)


class ProbeStatus(Enum):
    """Status of probe operations."""
    SUCCESS = "✅"
    FAILURE = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"


@dataclass
class SignalingProbeResult:
    """Results from signaling probe test."""
    test_name: str
    status: ProbeStatus
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class WebRTCSignalingProbe:
    """
    Probe tool for testing WebRTC signaling endpoints.
    
    Tests:
    1. Health check endpoint
    2. SDP offer/answer exchange
    3. ICE candidate exchange
    4. Peer connection status
    5. Cleanup/deletion
    """
    
    def __init__(self, base_url: str, timeout: int = 30, verbose: bool = False):
        """
        Initialize signaling probe.
        
        Args:
            base_url: Base URL of the API (e.g., https://dev.gmai.sa)
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.base_url = base_url.rstrip('/')
        self.signaling_base = f"{self.base_url}/api/v1/webrtc/voice"
        self.timeout = timeout
        self.verbose = verbose
        self.peer_id: Optional[str] = None
        self.results: List[SignalingProbeResult] = []
        
    def log(self, message: str, status: ProbeStatus = ProbeStatus.INFO):
        """Log message with status icon."""
        print(f"{status.value} {message}")
        
    def log_verbose(self, message: str):
        """Log verbose message."""
        if self.verbose:
            print(f"    {message}")
    
    def test_health_check(self) -> SignalingProbeResult:
        """Test health check endpoint."""
        self.log("Testing health check endpoint...", ProbeStatus.INFO)
        start_time = time.time()
        
        try:
            url = f"{self.signaling_base}/health"
            self.log_verbose(f"GET {url}")
            
            response = requests.get(url, timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.log_verbose(f"Response: {json.dumps(data, indent=2)}")
                
                result = SignalingProbeResult(
                    test_name="Health Check",
                    status=ProbeStatus.SUCCESS,
                    duration_ms=duration_ms,
                    details={
                        "status_code": response.status_code,
                        "response": data
                    }
                )
                self.log(f"Health check passed ({duration_ms:.0f}ms)", ProbeStatus.SUCCESS)
                return result
            else:
                result = SignalingProbeResult(
                    test_name="Health Check",
                    status=ProbeStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=f"Unexpected status code: {response.status_code}"
                )
                self.log(f"Health check failed: {response.status_code}", ProbeStatus.FAILURE)
                return result
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = SignalingProbeResult(
                test_name="Health Check",
                status=ProbeStatus.FAILURE,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            self.log(f"Health check failed: {e}", ProbeStatus.FAILURE)
            return result
    
    def test_sdp_offer_answer(self) -> SignalingProbeResult:
        """Test SDP offer/answer exchange."""
        self.log("Testing SDP offer/answer exchange...", ProbeStatus.INFO)
        start_time = time.time()
        
        try:
            url = f"{self.signaling_base}/offer"
            self.log_verbose(f"POST {url}")
            
            # Create mock SDP offer (simplified for testing)
            mock_offer = {
                "type": "offer",
                "sdp": (
                    "v=0\r\n"
                    "o=- 123456789 2 IN IP4 127.0.0.1\r\n"
                    "s=-\r\n"
                    "t=0 0\r\n"
                    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
                    "c=IN IP4 0.0.0.0\r\n"
                    "a=rtcp:9 IN IP4 0.0.0.0\r\n"
                    "a=ice-ufrag:test\r\n"
                    "a=ice-pwd:testpassword123456789012\r\n"
                    "a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\n"
                    "a=setup:actpass\r\n"
                    "a=rtpmap:111 opus/48000/2\r\n"
                )
            }
            
            payload = {
                "sdp_offer": mock_offer,
                "language": "ar",
                "client_capabilities": {
                    "echo_cancellation": True,
                    "noise_suppression": True,
                    "auto_gain_control": True
                }
            }
            
            self.log_verbose(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.log_verbose(f"Response: {json.dumps(data, indent=2)}")
                
                # Extract peer_id for subsequent tests
                self.peer_id = data.get("peer_id")
                
                # Validate response structure
                required_fields = ["sdp_answer", "peer_id", "session_id"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if missing_fields:
                    result = SignalingProbeResult(
                        test_name="SDP Offer/Answer",
                        status=ProbeStatus.WARNING,
                        duration_ms=duration_ms,
                        details={"response": data},
                        error_message=f"Missing fields: {missing_fields}"
                    )
                    self.log(f"SDP exchange incomplete: missing {missing_fields}", ProbeStatus.WARNING)
                    return result
                
                result = SignalingProbeResult(
                    test_name="SDP Offer/Answer",
                    status=ProbeStatus.SUCCESS,
                    duration_ms=duration_ms,
                    details={
                        "peer_id": self.peer_id,
                        "session_id": data.get("session_id"),
                        "sdp_answer_length": len(data.get("sdp_answer", {}).get("sdp", "")),
                        "ice_servers": data.get("ice_servers", [])
                    }
                )
                self.log(f"SDP exchange successful ({duration_ms:.0f}ms) - Peer ID: {self.peer_id}", 
                        ProbeStatus.SUCCESS)
                return result
            else:
                result = SignalingProbeResult(
                    test_name="SDP Offer/Answer",
                    status=ProbeStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=f"Status code: {response.status_code}, Body: {response.text}"
                )
                self.log(f"SDP exchange failed: {response.status_code}", ProbeStatus.FAILURE)
                return result
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = SignalingProbeResult(
                test_name="SDP Offer/Answer",
                status=ProbeStatus.FAILURE,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            self.log(f"SDP exchange failed: {e}", ProbeStatus.FAILURE)
            return result
    
    def test_ice_candidate_exchange(self) -> SignalingProbeResult:
        """Test ICE candidate exchange."""
        if not self.peer_id:
            result = SignalingProbeResult(
                test_name="ICE Candidate Exchange",
                status=ProbeStatus.FAILURE,
                duration_ms=0,
                error_message="No peer_id available (SDP exchange must succeed first)"
            )
            self.log("ICE test skipped: No peer_id", ProbeStatus.WARNING)
            return result
        
        self.log("Testing ICE candidate exchange...", ProbeStatus.INFO)
        start_time = time.time()
        
        try:
            url = f"{self.signaling_base}/ice"
            self.log_verbose(f"POST {url}")
            
            # Create mock ICE candidate
            mock_candidate = {
                "peer_id": self.peer_id,
                "candidate": "candidate:1 1 UDP 2130706431 192.168.1.100 54321 typ host",
                "sdp_mid": "0",
                "sdp_m_line_index": 0
            }
            
            self.log_verbose(f"Request payload: {json.dumps(mock_candidate, indent=2)}")
            
            response = requests.post(
                url,
                json=mock_candidate,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.log_verbose(f"Response: {json.dumps(data, indent=2)}")
                
                result = SignalingProbeResult(
                    test_name="ICE Candidate Exchange",
                    status=ProbeStatus.SUCCESS,
                    duration_ms=duration_ms,
                    details={"response": data}
                )
                self.log(f"ICE candidate accepted ({duration_ms:.0f}ms)", ProbeStatus.SUCCESS)
                return result
            else:
                result = SignalingProbeResult(
                    test_name="ICE Candidate Exchange",
                    status=ProbeStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=f"Status code: {response.status_code}, Body: {response.text}"
                )
                self.log(f"ICE candidate rejected: {response.status_code}", ProbeStatus.FAILURE)
                return result
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = SignalingProbeResult(
                test_name="ICE Candidate Exchange",
                status=ProbeStatus.FAILURE,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            self.log(f"ICE candidate exchange failed: {e}", ProbeStatus.FAILURE)
            return result
    
    def test_peer_status(self) -> SignalingProbeResult:
        """Test peer connection status endpoint."""
        if not self.peer_id:
            result = SignalingProbeResult(
                test_name="Peer Status",
                status=ProbeStatus.FAILURE,
                duration_ms=0,
                error_message="No peer_id available"
            )
            self.log("Status test skipped: No peer_id", ProbeStatus.WARNING)
            return result
        
        self.log("Testing peer status endpoint...", ProbeStatus.INFO)
        start_time = time.time()
        
        try:
            url = f"{self.signaling_base}/{self.peer_id}/status"
            self.log_verbose(f"GET {url}")
            
            response = requests.get(url, timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.log_verbose(f"Response: {json.dumps(data, indent=2)}")
                
                result = SignalingProbeResult(
                    test_name="Peer Status",
                    status=ProbeStatus.SUCCESS,
                    duration_ms=duration_ms,
                    details={
                        "peer_id": data.get("peer_id"),
                        "connection_state": data.get("connection_state"),
                        "ice_connection_state": data.get("ice_connection_state"),
                        "ice_gathering_state": data.get("ice_gathering_state")
                    }
                )
                self.log(f"Status retrieved ({duration_ms:.0f}ms): {data.get('connection_state')}", 
                        ProbeStatus.SUCCESS)
                return result
            else:
                result = SignalingProbeResult(
                    test_name="Peer Status",
                    status=ProbeStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=f"Status code: {response.status_code}"
                )
                self.log(f"Status retrieval failed: {response.status_code}", ProbeStatus.FAILURE)
                return result
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = SignalingProbeResult(
                test_name="Peer Status",
                status=ProbeStatus.FAILURE,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            self.log(f"Status retrieval failed: {e}", ProbeStatus.FAILURE)
            return result
    
    def test_cleanup(self) -> SignalingProbeResult:
        """Test peer connection cleanup/deletion."""
        if not self.peer_id:
            result = SignalingProbeResult(
                test_name="Cleanup",
                status=ProbeStatus.FAILURE,
                duration_ms=0,
                error_message="No peer_id available"
            )
            self.log("Cleanup test skipped: No peer_id", ProbeStatus.WARNING)
            return result
        
        self.log("Testing cleanup endpoint...", ProbeStatus.INFO)
        start_time = time.time()
        
        try:
            url = f"{self.signaling_base}/{self.peer_id}"
            self.log_verbose(f"DELETE {url}")
            
            response = requests.delete(url, timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.log_verbose(f"Response: {json.dumps(data, indent=2)}")
                
                result = SignalingProbeResult(
                    test_name="Cleanup",
                    status=ProbeStatus.SUCCESS,
                    duration_ms=duration_ms,
                    details={"response": data}
                )
                self.log(f"Cleanup successful ({duration_ms:.0f}ms)", ProbeStatus.SUCCESS)
                return result
            else:
                result = SignalingProbeResult(
                    test_name="Cleanup",
                    status=ProbeStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=f"Status code: {response.status_code}"
                )
                self.log(f"Cleanup failed: {response.status_code}", ProbeStatus.FAILURE)
                return result
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = SignalingProbeResult(
                test_name="Cleanup",
                status=ProbeStatus.FAILURE,
                duration_ms=duration_ms,
                error_message=str(e)
            )
            self.log(f"Cleanup failed: {e}", ProbeStatus.FAILURE)
            return result
    
    def run_all_tests(self) -> bool:
        """
        Run all signaling probe tests in sequence.
        
        Returns:
            True if all tests passed, False otherwise
        """
        print("\n" + "="*60)
        print("🔍 WebRTC Signaling Probe")
        print("="*60)
        print(f"Target URL: {self.base_url}")
        print(f"Timeout: {self.timeout}s")
        print(f"Verbose: {self.verbose}")
        print("="*60 + "\n")
        
        # Test sequence
        test_methods = [
            self.test_health_check,
            self.test_sdp_offer_answer,
            self.test_ice_candidate_exchange,
            self.test_peer_status,
            self.test_cleanup
        ]
        
        for test_method in test_methods:
            result = test_method()
            self.results.append(result)
            
            # Brief delay between tests
            time.sleep(0.5)
        
        # Print summary
        self._print_summary()
        
        # Return success if all tests passed
        return all(r.status == ProbeStatus.SUCCESS for r in self.results)
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == ProbeStatus.SUCCESS)
        failed_tests = sum(1 for r in self.results if r.status == ProbeStatus.FAILURE)
        warning_tests = sum(1 for r in self.results if r.status == ProbeStatus.WARNING)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Warnings: {warning_tests} ⚠️")
        print()
        
        # Detailed results
        for result in self.results:
            print(f"{result.status.value} {result.test_name}: {result.duration_ms:.0f}ms")
            if result.error_message:
                print(f"    Error: {result.error_message}")
        
        print("="*60)
        
        # Overall status
        if failed_tests == 0:
            if warning_tests == 0:
                print("✅ All tests passed!")
            else:
                print("⚠️  Tests passed with warnings")
        else:
            print("❌ Some tests failed")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WebRTC Signaling Probe Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test production server
  python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa
  
  # Test local development server
  python tools/webrtc_signaling_probe.py --url http://localhost:8000 --verbose
  
  # Test with custom timeout
  python tools/webrtc_signaling_probe.py --url https://dev.gmai.sa --timeout 60
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--test",
        choices=["health", "offer", "ice", "status", "cleanup", "all"],
        default="all",
        help="Run specific test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Create probe
    probe = WebRTCSignalingProbe(
        base_url=args.url,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Run tests
    if args.test == "all":
        success = probe.run_all_tests()
    else:
        test_map = {
            "health": probe.test_health_check,
            "offer": probe.test_sdp_offer_answer,
            "ice": probe.test_ice_candidate_exchange,
            "status": probe.test_peer_status,
            "cleanup": probe.test_cleanup
        }
        result = test_map[args.test]()
        probe.results.append(result)
        probe._print_summary()
        success = result.status == ProbeStatus.SUCCESS
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
