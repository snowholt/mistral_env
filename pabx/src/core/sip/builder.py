"""
SIP message builder
Constructs SIP messages and responses
"""

import random
import string
from typing import Dict, List, Optional
from datetime import datetime

from .types import SIPMethod, SIPResponse, SIP_VERSION
from .parser import SIPMessage


class SIPBuilder:
    """SIP message builder"""
    
    @staticmethod
    def generate_tag() -> str:
        """Generate random tag for From/To headers"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    
    @staticmethod
    def generate_branch() -> str:
        """Generate random branch parameter for Via header"""
        return 'z9hG4bK' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
    
    @staticmethod
    def generate_call_id() -> str:
        """Generate random Call-ID"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
    
    @staticmethod
    def build_response(
        request: SIPMessage,
        status_code: int,
        reason_phrase: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        body: str = "",
        to_tag: Optional[str] = None
    ) -> str:
        """
        Build SIP response from request
        
        Args:
            request: Original SIP request message
            status_code: Response status code
            reason_phrase: Response reason phrase (auto-generated if None)
            additional_headers: Additional headers to include
            body: Response body
            to_tag: Tag for To header (auto-generated if None and not in request)
            
        Returns:
            Formatted SIP response string
        """
        if reason_phrase is None:
            reason_phrase = SIPBuilder._get_reason_phrase(status_code)
        
        lines = []
        
        # Status line
        lines.append(f"{SIP_VERSION} {status_code} {reason_phrase}")
        
        # Via headers (copy from request)
        for via in request.via_headers:
            lines.append(f"Via: {via}")
        
        # From header (copy from request)
        from_hdr = request.from_header
        if from_hdr:
            lines.append(f"From: {from_hdr}")
        
        # To header (add tag if not present)
        to_hdr = request.to_header
        if to_hdr:
            # Check if tag already exists
            if ';tag=' not in to_hdr:
                if to_tag is None:
                    to_tag = SIPBuilder.generate_tag()
                to_hdr = f"{to_hdr};tag={to_tag}"
            lines.append(f"To: {to_hdr}")
        
        # Call-ID (copy from request)
        call_id = request.call_id
        if call_id:
            lines.append(f"Call-ID: {call_id}")
        
        # CSeq (copy from request)
        cseq = request.get_header("CSeq")
        if cseq:
            lines.append(f"CSeq: {cseq}")
        
        # Additional headers
        if additional_headers:
            for key, value in additional_headers.items():
                lines.append(f"{key}: {value}")
        
        # Content-Type and Content-Length
        if body:
            if additional_headers is None or 'Content-Type' not in additional_headers:
                lines.append("Content-Type: application/sdp")
            lines.append(f"Content-Length: {len(body)}")
        else:
            lines.append("Content-Length: 0")
        
        # Empty line before body
        lines.append("")
        
        # Body
        if body:
            lines.append(body)
        
        return '\r\n'.join(lines)
    
    @staticmethod
    def build_request(
        method: SIPMethod,
        request_uri: str,
        from_uri: str,
        to_uri: str,
        call_id: Optional[str] = None,
        cseq_number: int = 1,
        via_host: str = "127.0.0.1",
        via_port: int = 5060,
        contact: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        body: str = ""
    ) -> str:
        """
        Build SIP request
        
        Args:
            method: SIP method
            request_uri: Request URI
            from_uri: From URI
            to_uri: To URI
            call_id: Call-ID (auto-generated if None)
            cseq_number: CSeq sequence number
            via_host: Via host
            via_port: Via port
            contact: Contact header
            additional_headers: Additional headers
            body: Request body
            
        Returns:
            Formatted SIP request string
        """
        if call_id is None:
            call_id = SIPBuilder.generate_call_id()
        
        lines = []
        
        # Request line
        lines.append(f"{method.value} {request_uri} {SIP_VERSION}")
        
        # Via
        branch = SIPBuilder.generate_branch()
        lines.append(f"Via: {SIP_VERSION}/UDP {via_host}:{via_port};branch={branch}")
        
        # From
        from_tag = SIPBuilder.generate_tag()
        lines.append(f"From: <{from_uri}>;tag={from_tag}")
        
        # To
        lines.append(f"To: <{to_uri}>")
        
        # Call-ID
        lines.append(f"Call-ID: {call_id}")
        
        # CSeq
        lines.append(f"CSeq: {cseq_number} {method.value}")
        
        # Contact
        if contact:
            lines.append(f"Contact: <{contact}>")
        
        # Max-Forwards
        lines.append("Max-Forwards: 70")
        
        # User-Agent
        lines.append("User-Agent: BeautyAI-PABX/2.0")
        
        # Additional headers
        if additional_headers:
            for key, value in additional_headers.items():
                lines.append(f"{key}: {value}")
        
        # Content-Type and Content-Length
        if body:
            if additional_headers is None or 'Content-Type' not in additional_headers:
                lines.append("Content-Type: application/sdp")
            lines.append(f"Content-Length: {len(body)}")
        else:
            lines.append("Content-Length: 0")
        
        # Empty line before body
        lines.append("")
        
        # Body
        if body:
            lines.append(body)
        
        return '\r\n'.join(lines)
    
    @staticmethod
    def build_sdp(
        host: str,
        port: int,
        session_name: str = "BeautyAI PABX Session",
        codecs: Optional[List[int]] = None
    ) -> str:
        """
        Build SDP (Session Description Protocol) body
        
        Args:
            host: RTP host IP address
            port: RTP port
            session_name: Session name
            codecs: List of codec payload types (default: [0, 8, 101])
            
        Returns:
            SDP body string
        """
        if codecs is None:
            codecs = [0, 8, 101]  # PCMU, PCMA, telephone-event
        
        session_id = random.randint(100000, 999999)
        session_version = session_id
        
        lines = []
        
        # Version
        lines.append("v=0")
        
        # Origin
        lines.append(f"o=beautyai {session_id} {session_version} IN IP4 {host}")
        
        # Session name
        lines.append(f"s={session_name}")
        
        # Connection
        lines.append(f"c=IN IP4 {host}")
        
        # Time
        lines.append("t=0 0")
        
        # Media
        codec_str = ' '.join(str(c) for c in codecs)
        lines.append(f"m=audio {port} RTP/AVP {codec_str}")
        
        # RTP map attributes
        if 0 in codecs:
            lines.append("a=rtpmap:0 PCMU/8000")
        if 8 in codecs:
            lines.append("a=rtpmap:8 PCMA/8000")
        if 9 in codecs:
            lines.append("a=rtpmap:9 G722/8000")
        if 18 in codecs:
            lines.append("a=rtpmap:18 G729/8000")
        if 101 in codecs:
            lines.append("a=rtpmap:101 telephone-event/8000")
            lines.append("a=fmtp:101 0-16")
        
        # Media attributes
        lines.append("a=ptime:20")
        lines.append("a=sendrecv")
        
        return '\r\n'.join(lines)
    
    @staticmethod
    def _get_reason_phrase(status_code: int) -> str:
        """Get standard reason phrase for status code"""
        phrases = {
            100: "Trying",
            180: "Ringing",
            181: "Call Is Being Forwarded",
            182: "Queued",
            183: "Session Progress",
            200: "OK",
            202: "Accepted",
            300: "Multiple Choices",
            301: "Moved Permanently",
            302: "Moved Temporarily",
            305: "Use Proxy",
            380: "Alternative Service",
            400: "Bad Request",
            401: "Unauthorized",
            402: "Payment Required",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            406: "Not Acceptable",
            407: "Proxy Authentication Required",
            408: "Request Timeout",
            410: "Gone",
            413: "Request Entity Too Large",
            414: "Request-URI Too Long",
            415: "Unsupported Media Type",
            416: "Unsupported URI Scheme",
            420: "Bad Extension",
            421: "Extension Required",
            423: "Interval Too Brief",
            480: "Temporarily Unavailable",
            481: "Call/Transaction Does Not Exist",
            482: "Loop Detected",
            483: "Too Many Hops",
            484: "Address Incomplete",
            485: "Ambiguous",
            486: "Busy Here",
            487: "Request Terminated",
            488: "Not Acceptable Here",
            491: "Request Pending",
            493: "Undecipherable",
            500: "Server Internal Error",
            501: "Not Implemented",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Server Time-out",
            505: "Version Not Supported",
            513: "Message Too Large",
            600: "Busy Everywhere",
            603: "Decline",
            604: "Does Not Exist Anywhere",
            606: "Not Acceptable",
        }
        return phrases.get(status_code, "Unknown")
