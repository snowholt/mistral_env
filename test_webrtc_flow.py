#!/usr/bin/env python3
"""
Test WebRTC backend flow: offer -> answer -> ice candidate
This demonstrates the correct working flow for WebRTC signaling.
"""

import asyncio
import json
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

async def test_webrtc_flow():
    """Test complete WebRTC signaling flow."""
    base_url = "http://192.168.100.39:8000"
    
    print("🧪 Testing WebRTC Backend Flow")
    print("=" * 50)
    
    # Create RTCPeerConnection to generate real offer
    pc = RTCPeerConnection()
    
    # Add transceiver for audio (required for WebRTC)
    pc.addTransceiver("audio", direction="sendrecv")
    
    # Create real SDP offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    print(f"📤 Generated SDP Offer (first 100 chars):")
    print(f"   {offer.sdp[:100]}...")
    print()
    
    # Step 1: Send offer to backend
    async with aiohttp.ClientSession() as session:
        offer_data = {
            "sdp": offer.sdp,
            "type": "offer",
            "language": "ar"
        }
        
        print("🔄 Step 1: Sending offer to backend...")
        async with session.post(
            f"{base_url}/api/v1/webrtc/voice/offer",
            json=offer_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                answer_data = await response.json()
                peer_id = answer_data["peer_id"]
                print(f"✅ Offer successful! peer_id: {peer_id}")
                print(f"   Answer SDP length: {len(answer_data['sdp'])} chars")
                print()
                
                # Step 2: Test ICE endpoint with valid peer_id
                print("🔄 Step 2: Testing ICE endpoint with valid peer_id...")
                ice_data = {
                    "peer_id": peer_id,
                    "candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host",
                    "sdp_mid": "0",
                    "sdp_m_line_index": 0
                }
                
                async with session.post(
                    f"{base_url}/api/v1/webrtc/voice/ice",
                    json=ice_data,
                    headers={"Content-Type": "application/json"}
                ) as ice_response:
                    if ice_response.status == 200:
                        ice_result = await ice_response.json()
                        print(f"✅ ICE candidate accepted!")
                        print(f"   Response: {ice_result}")
                        print()
                    else:
                        ice_error = await ice_response.text()
                        print(f"❌ ICE failed ({ice_response.status}): {ice_error}")
                        print()
                
                # Step 3: Test ICE with invalid peer_id
                print("🔄 Step 3: Testing ICE endpoint with invalid peer_id...")
                invalid_ice_data = {
                    "peer_id": "invalid_peer_id",
                    "candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 54321 typ host",
                    "sdp_mid": "0",
                    "sdp_m_line_index": 0
                }
                
                async with session.post(
                    f"{base_url}/api/v1/webrtc/voice/ice",
                    json=invalid_ice_data,
                    headers={"Content-Type": "application/json"}
                ) as invalid_ice_response:
                    print(f"📊 Invalid peer_id response: {invalid_ice_response.status}")
                    if invalid_ice_response.status == 404:
                        print("✅ Correctly returns 404 for invalid peer_id")
                    else:
                        error_text = await invalid_ice_response.text()
                        print(f"❌ Unexpected response: {error_text}")
                    print()
                
                # Step 4: Cleanup
                print("🔄 Step 4: Cleaning up connection...")
                async with session.delete(f"{base_url}/api/v1/webrtc/voice/{peer_id}") as cleanup_response:
                    if cleanup_response.status == 200:
                        cleanup_result = await cleanup_response.json()
                        print(f"✅ Cleanup successful: {cleanup_result['message']}")
                    else:
                        print(f"⚠️  Cleanup response: {cleanup_response.status}")
                
            else:
                error_text = await response.text()
                print(f"❌ Offer failed ({response.status}): {error_text}")
    
    # Close peer connection
    await pc.close()
    
    print("\n" + "=" * 50)
    print("🎯 Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_webrtc_flow())