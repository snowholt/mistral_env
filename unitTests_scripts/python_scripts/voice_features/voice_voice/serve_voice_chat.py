#!/usr/bin/env python3
"""
Simple HTTP Server for BeautyAI WebSocket Voice Chat.

This serves the websocket_voice_chat.html file via HTTP so it can properly
connect to the WebSocket endpoint on the remote machine.
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

def serve_voice_chat(port=8080):
    """
    Serve the voice chat HTML file via HTTP server.
    
    Args:
        port: Port to serve on (default: 8080)
    """
    # Change to the directory containing the HTML file
    html_dir = Path(__file__).parent
    os.chdir(html_dir)
    
    # Create HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
        print(f"🌐 Serving BeautyAI Voice Chat at:")
        print(f"   Local: http://localhost:{port}/websocket_voice_chat.html")
        print(f"   Remote: http://dev.gmai.sa:{port}/websocket_voice_chat.html")
        print(f"")
        print(f"📡 WebSocket will connect to: ws://dev.gmai.sa:8000/ws/voice-conversation")
        print(f"")
        print(f"🎤 Open the URL above in your browser to use voice chat!")
        print(f"   Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    serve_voice_chat()
