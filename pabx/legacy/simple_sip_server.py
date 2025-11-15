#!/usr/bin/env python3
"""
Simple SIP test server for HT813 testing
Accepts registrations and establishes basic calls
Enhanced with detailed logging
"""

import socket
import threading
import time
from datetime import datetime

class SimpleSIPServer:
    def __init__(self, host='0.0.0.0', port=5060):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.registered_users = {}
        self.active_calls = {}
        self.call_counter = 0
        
        print("=" * 80)
        print(f"🎙️  SIMPLE SIP SERVER - ENHANCED LOGGING")
        print("=" * 80)
        print(f"📍 Listening on: {host}:{port}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Features: REGISTER, INVITE, ACK, BYE, OPTIONS")
        print(f"📊 Logging: Enhanced with detailed message tracking")
        print("=" * 80)
        print(f"💡 Press Ctrl+C to stop")
        print()
        print("🔊 WAITING FOR REQUESTS...")
        print()
    
    def log_timestamp(self):
        """Get formatted timestamp for logs"""
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    def send_response(self, addr, response, msg_type="Response"):
        """Send SIP response with detailed logging"""
        timestamp = self.log_timestamp()
        response_code = response.split()[1] if len(response.split()) > 1 else "???"
        response_text = ' '.join(response.split()[0:3])
        
        self.sock.sendto(response.encode(), addr)
        
        print(f"[{timestamp}] 📤 SENT {msg_type} → {addr[0]}:{addr[1]}")
        print(f"            ✉️  {response_text}")
        print(f"            📦 Size: {len(response)} bytes")
        print()
    
    def handle_register(self, data, addr):
        """Handle SIP REGISTER request"""
        timestamp = self.log_timestamp()
        lines = data.split('\r\n')
        
        # Extract basic info
        user_id = None
        call_id = None
        cseq = None
        via = None
        contact = None
        expires = "3600"
        user_agent = None
        
        print(f"[{timestamp}] 📋 PROCESSING REGISTER")
        print(f"            📍 From: {addr[0]}:{addr[1]}")
        
        for line in lines:
            if line.startswith('From:') or line.startswith('f:'):
                # Extract user
                if 'sip:' in line:
                    user_id = line.split('sip:')[1].split('@')[0]
                    print(f"            👤 User: {user_id}")
            elif line.startswith('Call-ID:') or line.startswith('i:'):
                call_id = line.split(':', 1)[1].strip()
                print(f"            🔑 Call-ID: {call_id[:30]}...")
            elif line.startswith('CSeq:'):
                cseq = line.split(':', 1)[1].strip()
                print(f"            📊 CSeq: {cseq}")
            elif line.startswith('Via:') or line.startswith('v:'):
                via = line
            elif line.startswith('Contact:') or line.startswith('m:'):
                contact = line
                print(f"            📞 Contact: {contact[:60]}...")
            elif line.startswith('Expires:'):
                expires = line.split(':', 1)[1].strip()
                print(f"            ⏰ Expires: {expires}s")
            elif line.startswith('User-Agent:'):
                user_agent = line.split(':', 1)[1].strip()
                print(f"            🔧 User-Agent: {user_agent}")
        
        if user_id:
            self.registered_users[user_id] = {
                'addr': addr,
                'time': datetime.now(),
                'contact': contact,
                'user_agent': user_agent
            }
            print(f"            ✅ Registration accepted!")
            print(f"            💾 Stored in registry: {len(self.registered_users)} users total")
        else:
            print(f"            ⚠️  No user ID found!")
        
        # Send 200 OK
        response = f"""SIP/2.0 200 OK
{via}
From: <sip:{user_id}@{self.host}>
To: <sip:{user_id}@{self.host}>;tag=as7d8f9g
Call-ID: {call_id}
CSeq: {cseq}
{contact}
Expires: {expires}
Server: SimpleSIP/1.0
Content-Length: 0

"""
        self.send_response(addr, response, "200 OK (REGISTER)")
        print()
    
    def handle_invite(self, data, addr):
        """Handle SIP INVITE request"""
        timestamp = self.log_timestamp()
        lines = data.split('\r\n')
        
        self.call_counter += 1
        call_num = self.call_counter
        
        call_id = None
        cseq = None
        via = None
        from_header = None
        to_header = None
        from_user = None
        to_user = None
        contact = None
        sdp_body = ""
        rtp_port = 5004  # Default
        codec_list = []
        user_agent = None
        
        print("=" * 80)
        print(f"[{timestamp}] 📞 INCOMING CALL #{call_num}")
        print("=" * 80)
        print(f"            📍 Source: {addr[0]}:{addr[1]}")
        
        # Parse headers and SDP
        in_sdp = False
        for line in lines:
            if line.startswith('Call-ID:'):
                call_id = line.split(':', 1)[1].strip()
                print(f"            🔑 Call-ID: {call_id[:40]}...")
            elif line.startswith('CSeq:'):
                cseq = line.split(':', 1)[1].strip()
                print(f"            📊 CSeq: {cseq}")
            elif line.startswith('Via:'):
                via = line
            elif line.startswith('From:'):
                from_header = line
                if 'sip:' in line:
                    from_user = line.split('sip:')[1].split('@')[0]
                    print(f"            👤 From: {from_user}")
            elif line.startswith('To:'):
                to_header = line
                if 'sip:' in line:
                    to_user = line.split('sip:')[1].split('@')[0]
                    print(f"            📲 To: {to_user}")
            elif line.startswith('Contact:'):
                contact = line
            elif line.startswith('User-Agent:'):
                user_agent = line.split(':', 1)[1].strip()
                print(f"            🔧 User-Agent: {user_agent}")
            elif line.startswith('m=audio'):
                # Extract RTP port from SDP
                parts = line.split()
                if len(parts) >= 2:
                    rtp_port = int(parts[1])
                    print(f"            🎤 Client RTP Port: {rtp_port}")
                # Extract codec list
                if len(parts) >= 4:
                    codec_list = parts[3:]
                    print(f"            🎵 Offered Codecs: {' '.join(codec_list)}")
            elif line.startswith('a=rtpmap:'):
                codec_info = line.split(':', 1)[1].strip()
                print(f"            🎶 Codec: {codec_info}")
        
        # Store call info
        if call_id:
            self.active_calls[call_id] = {
                'call_num': call_num,
                'from': from_user,
                'to': to_user,
                'addr': addr,
                'rtp_port': rtp_port,
                'start_time': datetime.now(),
                'status': 'INVITING'
            }
        
        print()
        print(f"            ⏩ Sending call progress responses...")
        
        # Send 100 Trying
        trying = f"""SIP/2.0 100 Trying
{via}
{from_header}
{to_header}
Call-ID: {call_id}
CSeq: {cseq}
Server: SimpleSIP/1.0
Content-Length: 0

"""
        self.send_response(addr, trying, "100 Trying")
        
        # Send 180 Ringing
        time.sleep(0.1)
        ringing = f"""SIP/2.0 180 Ringing
{via}
{from_header}
{to_header};tag=ring123
Call-ID: {call_id}
CSeq: {cseq}
Server: SimpleSIP/1.0
Contact: <sip:test@{self.host}:5060>
Content-Length: 0

"""
        self.send_response(addr, ringing, "180 Ringing")
        
        # Send 200 OK with SDP (use the client's IP for RTP)
        time.sleep(0.5)
        sdp = f"""v=0
o=root 123456 123456 IN IP4 {self.host}
s=TestCall
c=IN IP4 {self.host}
t=0 0
m=audio 12000 RTP/AVP 0 8 101
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:101 telephone-event/8000
a=fmtp:101 0-16
a=ptime:20
a=sendrecv
"""
        
        ok = f"""SIP/2.0 200 OK
{via}
{from_header}
{to_header};tag=ok456
Call-ID: {call_id}
CSeq: {cseq}
Contact: <sip:test@{self.host}:5060>
Supported: timer, replaces
Server: SimpleSIP/1.0
Content-Type: application/sdp
Content-Length: {len(sdp)}

{sdp}"""
        self.send_response(addr, ok, "200 OK (INVITE)")
        
        if call_id and call_id in self.active_calls:
            self.active_calls[call_id]['status'] = 'ANSWERED'
        
        print(f"[{self.log_timestamp()}] ✅ CALL #{call_num} ANSWERED")
        print(f"            🎤 Server RTP Port: 12000")
        print(f"            📡 Client RTP Port: {rtp_port}")
        print(f"            🔄 Waiting for ACK to complete call setup...")
        print(f"            💡 RTP packets should flow between:")
        print(f"               • {addr[0]}:{rtp_port} ↔ {self.host}:12000")
        print("=" * 80)
        print()
    
    def handle_request(self, data, addr):
        """Handle incoming SIP request"""
        try:
            timestamp = self.log_timestamp()
            lines = data.split('\r\n')
            request_line = lines[0]
            method = request_line.split()[0] if lines else "UNKNOWN"
            
            # Extract Call-ID for correlation
            call_id = None
            for line in lines:
                if line.startswith('Call-ID:'):
                    call_id = line.split(':', 1)[1].strip()
                    break
            
            print(f"[{timestamp}] 📨 RECEIVED: {method}")
            print(f"            📍 From: {addr[0]}:{addr[1]}")
            if call_id:
                call_id_short = call_id[:40] + "..." if len(call_id) > 40 else call_id
                print(f"            🔑 Call-ID: {call_id_short}")
            
            if request_line.startswith('REGISTER'):
                self.handle_register(data, addr)
                
            elif request_line.startswith('INVITE'):
                self.handle_invite(data, addr)
                
            elif request_line.startswith('ACK'):
                # Find associated call
                call_info = self.active_calls.get(call_id) if call_id else None
                if call_info:
                    call_info['status'] = 'ACTIVE'
                    call_num = call_info['call_num']
                    duration = (datetime.now() - call_info['start_time']).total_seconds()
                    
                    print(f"            ✅ ACK received for Call #{call_num}")
                    print(f"            🎉 CALL #{call_num} IS NOW ACTIVE!")
                    print(f"            ⏱️  Setup time: {duration:.2f}s")
                    print(f"            🎤 RTP should be flowing now!")
                    print(f"            📡 Monitor port {call_info['rtp_port']} for RTP packets")
                else:
                    print(f"            ✅ ACK received (no call tracking)")
                print("=" * 80)
                print()
                
            elif request_line.startswith('BYE'):
                # Find associated call
                call_info = self.active_calls.get(call_id) if call_id else None
                if call_info:
                    duration = (datetime.now() - call_info['start_time']).total_seconds()
                    call_num = call_info['call_num']
                    
                    print(f"            📵 BYE received for Call #{call_num}")
                    print(f"            ⏱️  Call duration: {duration:.2f}s")
                    print(f"            👋 Ending call...")
                    
                    # Remove from active calls
                    del self.active_calls[call_id]
                else:
                    print(f"            📵 BYE received")
                
                # Send 200 OK for BYE
                lines = data.split('\r\n')
                call_id_hdr = via = cseq = from_hdr = to_hdr = None
                for line in lines:
                    if 'Call-ID:' in line: call_id_hdr = line.split(':', 1)[1].strip()
                    if 'CSeq:' in line: cseq = line.split(':', 1)[1].strip()
                    if 'Via:' in line: via = line
                    if 'From:' in line: from_hdr = line
                    if 'To:' in line: to_hdr = line
                
                response = f"""SIP/2.0 200 OK
{via}
{from_hdr}
{to_hdr}
Call-ID: {call_id_hdr}
CSeq: {cseq}
Server: SimpleSIP/1.0
Content-Length: 0

"""
                self.send_response(addr, response, "200 OK (BYE)")
                
                print(f"[{self.log_timestamp()}] ✅ Call #{call_num if call_info else '?'} ended gracefully")
                print("=" * 80)
                print()
                
            elif request_line.startswith('OPTIONS'):
                print(f"            💓 Keepalive OPTIONS ping")
                
                # Send 200 OK for keepalive
                lines = data.split('\r\n')
                call_id_hdr = via = cseq = None
                for line in lines:
                    if 'Call-ID:' in line: call_id_hdr = line.split(':', 1)[1].strip()
                    if 'CSeq:' in line: cseq = line.split(':', 1)[1].strip()
                    if 'Via:' in line: via = line
                
                response = f"""SIP/2.0 200 OK
{via}
Call-ID: {call_id_hdr}
CSeq: {cseq}
Allow: INVITE, ACK, CANCEL, BYE, OPTIONS
Accept: application/sdp
Server: SimpleSIP/1.0
Content-Length: 0

"""
                self.send_response(addr, response, "200 OK (OPTIONS)")
                print()
            
            elif request_line.startswith('CANCEL'):
                print(f"            ❌ CANCEL received - Call cancelled")
                # Send 200 OK for CANCEL
                lines = data.split('\r\n')
                call_id_hdr = via = cseq = None
                for line in lines:
                    if 'Call-ID:' in line: call_id_hdr = line.split(':', 1)[1].strip()
                    if 'CSeq:' in line: cseq = line.split(':', 1)[1].strip()
                    if 'Via:' in line: via = line
                
                response = f"""SIP/2.0 200 OK
{via}
Call-ID: {call_id_hdr}
CSeq: {cseq}
Server: SimpleSIP/1.0
Content-Length: 0

"""
                self.send_response(addr, response, "200 OK (CANCEL)")
                
                # Remove from active calls if exists
                if call_id and call_id in self.active_calls:
                    del self.active_calls[call_id]
                print()
            else:
                print(f"            ⚠️  Unknown method: {method}")
                print()
            
        except Exception as e:
            print(f"❌ ERROR handling request: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    def run(self):
        """Run the server"""
        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                data = data.decode('utf-8', errors='ignore')
                
                # Handle in a thread to avoid blocking
                threading.Thread(target=self.handle_request, args=(data, addr), daemon=True).start()
                
        except KeyboardInterrupt:
            print("\n")
            print("=" * 80)
            print("⏹️  SERVER SHUTDOWN")
            print("=" * 80)
            print(f"⏰ Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        except Exception as e:
            print(f"\n❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sock.close()
            
            # Print final statistics
            print()
            print("=" * 80)
            print("📊 SESSION STATISTICS")
            print("=" * 80)
            print(f"📞 Total calls handled: {self.call_counter}")
            print(f"👥 Registered users: {len(self.registered_users)}")
            
            if self.registered_users:
                print()
                print("📋 Registered Users:")
                for user, info in self.registered_users.items():
                    reg_time = info['time'].strftime('%H:%M:%S')
                    print(f"   • {user} from {info['addr'][0]}:{info['addr'][1]} (registered at {reg_time})")
            
            if self.active_calls:
                print()
                print("⚠️  Active calls at shutdown:")
                for cid, call in self.active_calls.items():
                    duration = (datetime.now() - call['start_time']).total_seconds()
                    print(f"   • Call #{call['call_num']}: {call['from']} → {call['to']} ({duration:.1f}s)")
            
            print("=" * 80)
            print("👋 Goodbye!")
            print()

if __name__ == '__main__':
    server = SimpleSIPServer()
    server.run()
