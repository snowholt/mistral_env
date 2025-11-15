"""
SIP message parser
Parses raw SIP messages into structured format
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .types import SIPMethod, SIPResponse, COMPACT_HEADERS, SIP_VERSION


@dataclass
class SIPMessage:
    """Parsed SIP message structure"""
    
    # Request line / Status line
    is_request: bool
    method: Optional[SIPMethod] = None
    request_uri: Optional[str] = None
    status_code: Optional[int] = None
    reason_phrase: Optional[str] = None
    sip_version: str = SIP_VERSION
    
    # Headers (normalized to canonical form)
    headers: Dict[str, List[str]] = field(default_factory=dict)
    
    # Body
    body: str = ""
    
    # Raw message
    raw: str = ""
    
    def get_header(self, name: str, index: int = 0) -> Optional[str]:
        """Get header value by name (supports compact forms)"""
        # Normalize header name
        normalized = COMPACT_HEADERS.get(name, name)
        
        # Ensure index is an integer (in case string is passed)
        index = int(index) if not isinstance(index, int) else index
        
        if normalized in self.headers:
            values = self.headers[normalized]
            if index < len(values):
                return values[index]
        return None
    
    def get_all_headers(self, name: str) -> List[str]:
        """Get all values for a header"""
        normalized = COMPACT_HEADERS.get(name, name)
        return self.headers.get(normalized, [])
    
    def set_header(self, name: str, value: str, append: bool = False):
        """Set or append header value"""
        normalized = COMPACT_HEADERS.get(name, name)
        
        if append and normalized in self.headers:
            self.headers[normalized].append(value)
        else:
            self.headers[normalized] = [value]
    
    @property
    def call_id(self) -> Optional[str]:
        """Get Call-ID"""
        return self.get_header("Call-ID")
    
    @property
    def cseq(self) -> Optional[Tuple[int, str]]:
        """Get CSeq as (sequence_number, method) tuple"""
        cseq_str = self.get_header("CSeq")
        if cseq_str:
            parts = cseq_str.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    return (int(parts[0]), parts[1])
                except ValueError:
                    pass
        return None
    
    @property
    def from_header(self) -> Optional[str]:
        """Get From header"""
        return self.get_header("From")
    
    @property
    def to_header(self) -> Optional[str]:
        """Get To header"""
        return self.get_header("To")
    
    @property
    def via_headers(self) -> List[str]:
        """Get all Via headers"""
        return self.get_all_headers("Via")
    
    @property
    def contact(self) -> Optional[str]:
        """Get Contact header"""
        return self.get_header("Contact")
    
    @property
    def content_type(self) -> Optional[str]:
        """Get Content-Type"""
        return self.get_header("Content-Type")
    
    @property
    def content_length(self) -> int:
        """Get Content-Length"""
        cl = self.get_header("Content-Length")
        if cl:
            try:
                return int(cl)
            except ValueError:
                pass
        return 0
    
    def extract_sdp(self) -> Optional[Dict[str, any]]:
        """Extract and parse SDP body if present"""
        if self.content_type != "application/sdp":
            return None
        
        if not self.body:
            return None
        
        return parse_sdp(self.body)


class SIPParser:
    """SIP message parser"""
    
    # Regex patterns
    REQUEST_LINE_PATTERN = re.compile(
        r'^(REGISTER|INVITE|ACK|BYE|CANCEL|OPTIONS|INFO|PRACK|UPDATE|REFER|SUBSCRIBE|NOTIFY|MESSAGE)\s+'
        r'(sip:[^\s]+)\s+'
        r'SIP/(\d\.\d)\s*$',
        re.IGNORECASE
    )
    
    STATUS_LINE_PATTERN = re.compile(
        r'^SIP/(\d\.\d)\s+(\d{3})\s+(.+?)\s*$',
        re.IGNORECASE
    )
    
    HEADER_PATTERN = re.compile(
        r'^([^:]+):\s*(.*)$'
    )
    
    @staticmethod
    def parse(data: str) -> Optional[SIPMessage]:
        """
        Parse raw SIP message
        
        Args:
            data: Raw SIP message string
            
        Returns:
            SIPMessage object or None if parsing fails
        """
        if not data or not data.strip():
            return None
        
        lines = data.split('\r\n')
        if not lines:
            return None
        
        # Parse first line (request or status)
        first_line = lines[0]
        msg = SIPParser._parse_first_line(first_line)
        if not msg:
            return None
        
        msg.raw = data
        
        # Parse headers
        body_start = SIPParser._parse_headers(lines[1:], msg)
        
        # Parse body if present
        if body_start < len(lines):
            msg.body = '\r\n'.join(lines[body_start:])
        
        return msg
    
    @staticmethod
    def _parse_first_line(line: str) -> Optional[SIPMessage]:
        """Parse request line or status line"""
        # Try request line
        match = SIPParser.REQUEST_LINE_PATTERN.match(line)
        if match:
            method_str = match.group(1).upper()
            try:
                method = SIPMethod(method_str)
            except ValueError:
                method = None
            
            return SIPMessage(
                is_request=True,
                method=method,
                request_uri=match.group(2),
                sip_version=f"SIP/{match.group(3)}"
            )
        
        # Try status line
        match = SIPParser.STATUS_LINE_PATTERN.match(line)
        if match:
            return SIPMessage(
                is_request=False,
                status_code=int(match.group(2)),
                reason_phrase=match.group(3),
                sip_version=f"SIP/{match.group(1)}"
            )
        
        return None
    
    @staticmethod
    def _parse_headers(lines: List[str], msg: SIPMessage) -> int:
        """
        Parse headers from lines
        
        Returns:
            Index where body starts
        """
        i = 0
        current_header = None
        current_value = None
        
        while i < len(lines):
            line = lines[i]
            
            # Empty line indicates end of headers
            if not line or line.strip() == '':
                i += 1
                break
            
            # Check for continuation (line starts with whitespace)
            if line[0] in (' ', '\t'):
                if current_header:
                    current_value += ' ' + line.strip()
                i += 1
                continue
            
            # Save previous header if exists
            if current_header:
                msg.set_header(current_header, current_value, append=True)
            
            # Parse new header
            match = SIPParser.HEADER_PATTERN.match(line)
            if match:
                current_header = match.group(1).strip()
                current_value = match.group(2).strip()
                
                # Expand compact form
                current_header = COMPACT_HEADERS.get(current_header, current_header)
            else:
                current_header = None
                current_value = None
            
            i += 1
        
        # Save last header
        if current_header:
            msg.set_header(current_header, current_value, append=True)
        
        return i
    
    @staticmethod
    def extract_user_from_uri(uri: str) -> Optional[str]:
        """Extract username from SIP URI"""
        # sip:user@domain -> user
        if 'sip:' in uri:
            parts = uri.split('sip:')[1].split('@')
            if parts:
                return parts[0].strip('<>')
        return None
    
    @staticmethod
    def extract_tag(header: str) -> Optional[str]:
        """Extract tag parameter from From/To header"""
        if ';tag=' in header:
            tag_part = header.split(';tag=')[1]
            return tag_part.split(';')[0].split('>')[0].strip()
        return None
    
    @staticmethod
    def extract_parameter(header: str, param_name: str) -> Optional[str]:
        """Extract parameter from header"""
        pattern = rf';{param_name}=([^;>\s]+)'
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            return match.group(1)
        return None


def parse_sdp(sdp_body: str) -> Dict[str, any]:
    """
    Parse SDP (Session Description Protocol) body
    
    Returns a dictionary with parsed SDP fields
    """
    sdp = {
        'version': None,
        'origin': {},
        'session_name': None,
        'connection': {},
        'time': {},
        'media': [],
        'attributes': []
    }
    
    current_media = None
    
    for line in sdp_body.split('\r\n'):
        if not line or '=' not in line:
            continue
        
        field_type = line[0]
        value = line[2:].strip()
        
        if field_type == 'v':  # Version
            sdp['version'] = value
            
        elif field_type == 'o':  # Origin
            parts = value.split()
            if len(parts) >= 6:
                sdp['origin'] = {
                    'username': parts[0],
                    'session_id': parts[1],
                    'session_version': parts[2],
                    'network_type': parts[3],
                    'address_type': parts[4],
                    'address': parts[5]
                }
        
        elif field_type == 's':  # Session name
            sdp['session_name'] = value
        
        elif field_type == 'c':  # Connection
            parts = value.split()
            if len(parts) >= 3:
                conn = {
                    'network_type': parts[0],
                    'address_type': parts[1],
                    'address': parts[2]
                }
                if current_media:
                    current_media['connection'] = conn
                else:
                    sdp['connection'] = conn
        
        elif field_type == 't':  # Time
            parts = value.split()
            if len(parts) >= 2:
                sdp['time'] = {
                    'start': parts[0],
                    'stop': parts[1]
                }
        
        elif field_type == 'm':  # Media
            parts = value.split()
            if len(parts) >= 4:
                current_media = {
                    'type': parts[0],
                    'port': int(parts[1]),
                    'protocol': parts[2],
                    'formats': parts[3:],
                    'attributes': []
                }
                sdp['media'].append(current_media)
        
        elif field_type == 'a':  # Attribute
            attr = {'value': value}
            
            # Parse attribute key/value
            if ':' in value:
                key, val = value.split(':', 1)
                attr = {'key': key, 'value': val}
            else:
                attr = {'key': value, 'value': None}
            
            if current_media:
                current_media['attributes'].append(attr)
            else:
                sdp['attributes'].append(attr)
    
    return sdp
