
import requests
import json

def test_offer():
    url = "http://localhost:8000/api/v1/webrtc/voice/offer"
    payload = {
        "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\na=mid:0\r\na=sendrecv\r\na=msid:audio audio\r\na=rtcp-mux\r\n",
        "type": "offer",
        "language": "ar"
    }
    
    try:
        print(f"Sending POST to {url}...")
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_offer()
