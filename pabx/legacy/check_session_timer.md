# Session Timer Issue - Final Fix

## Problem Identified:
The XML configuration shows session timer is PARTIALLY disabled:
- ✅ Caller Request Timer = 0 (No) 
- ✅ Callee Request Timer = 0 (No)
- ✅ Force Timer = 0 (No)
- ⚠️ Session Expiration = 180 seconds (still set!)

## The Real Issue:
The call is RINGING but disconnecting. This could be:

### Option 1: Ring Timeout (Most Likely!)
- **Ring Timeout** is set to 60 seconds (default)
- If phone rings but isn't answered within 60 seconds, call ends
- **Solution**: Pick up the phone IMMEDIATELY when it rings!

### Option 2: Session Timer Still Active
- Even though request timers are disabled, expiration is still 180s
- **Solution**: Need to verify "Enable Session Timer" is set to "No" in web UI

### Option 3: No Answer / Phone Issue
- Phone is ringing but not being picked up
- Or phone hardware issue
- **Solution**: Ensure phone is actually picked up during ring

## Next Steps:

### TEST 1: Answer the call IMMEDIATELY
1. Have phone ready in hand
2. Make call (dial 2001)
3. **Pick up phone IMMEDIATELY when it starts ringing**
4. Talk for 30+ seconds
5. Check if it stays connected

### TEST 2: Check Ring Timeout
In HT813 web UI → FXS Port:
- Find "Ring Timeout" (should be 60 seconds)
- This is how long it rings before giving up
- **Make sure to answer within 60 seconds!**

### TEST 3: Verify Session Timer Completely Off
In HT813 web UI → FXS Port:
- Look for "Enable Session Timer" dropdown
- Make absolutely sure it says "No" (not "Yes")
- If it says "Yes", change to "No" and Apply

## Key Question:
**Is the phone being ANSWERED (picked up) when it rings?**
- If YES → Session timer issue
- If NO → That's why it disconnects (ring timeout)!

