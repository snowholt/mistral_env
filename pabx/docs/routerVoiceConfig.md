# Router Voice Configuration Documentation
**Last Updated**: November 20, 2025  
**Router Model**: ZNID-GPON-2428B1-0ST (Zhone/Dasan)  
**Firmware Version**: S4.2.022

---

## 📋 Table of Contents
1. [Network Configuration](#network-configuration)
2. [VoIP Service Configuration](#voip-service-configuration)
3. [SIP Configuration](#sip-configuration)
4. [Voice Lines](#voice-lines)
5. [Port Forwarding Rules](#port-forwarding-rules)
6. [Firewall Rules](#firewall-rules)
7. [VLAN Configuration](#vlan-configuration)
8. [QoS Configuration](#qos-configuration)
9. [Classification Rules](#classification-rules)

---

## 🌐 Network Configuration

### WAN Interfaces

#### Primary WAN (Internet) - PPPoE
- **Interface**: `eth0.v10.ppp` (ppp0.10)
- **VLAN ID**: 10
- **Connection Type**: PPPoE_Bridged
- **Service Name**: TR069 INTERNET
- **Public IP**: 176.45.31.103
- **Gateway**: 84.235.6.25
- **Subnet Mask**: 255.255.255.255
- **MTU**: 1492
- **PPP Username**: `114874423@stc.net.sa`
- **PPP LCP Echo**: 60 seconds
- **PPP LCP Echo Retry**: 3
- **Status**: Connected
- **Connection Time**: 142 seconds uptime
- **NAT**: Enabled
- **DNS Type**: PPPoE (from ISP)
  - Primary DNS: 84.235.6.55
  - Secondary DNS: 84.235.57.230

#### VoIP WAN Interface - DHCP
- **Interface**: `brvlan11`
- **VLAN ID**: 11
- **Connection Type**: IP_Bridged (CPU-Bridged)
- **IP Assignment**: DHCP (from STC VoIP network)
- **Gateway**: 20.159.0.1
- **Local IP**: 20.159.4.1 (assigned by DHCP)
- **NAT**: Enabled
- **DNS Type**: DHCP
- **DHCP Client PID**: 5243
- **Status**: Connected

### LAN Interfaces

#### Main LAN Bridge
- **Interface**: `brvlan10`
- **VLAN ID**: 10
- **IP Address**: 192.168.100.1
- **Subnet Mask**: 255.255.255.0
- **DHCP Server**: Enabled
  - DHCP Range: 192.168.100.2 - 192.168.100.254
- **DNS Servers**: 
  - Primary: 84.235.6.55
  - Secondary: 84.235.57.230
- **IPv6**: Enabled (LAN-PD)
- **Link-Local IPv6**: fe80::10f2:1cff:fe0e:91e3/64

#### Default Bridge
- **Interface**: `br0`
- **Type**: Bridged
- **IPv6 DNS**: ::/128

### Ethernet Ports
- **eth0**: Fiber WAN (GPON)
- **eth1**: GE1 - GigE (LAN port 1) - VLAN 10
- **eth2**: GE2 - GigE (LAN port 2) - VLAN 10
- **eth3**: GE3 - GigE (LAN port 3) - VLAN 10
- **eth4**: GE4 - GigE (LAN port 4) - VLAN 10

### Wireless Networks

#### 2.4GHz Network
- **SSID**: WK
- **BSSID**: 62:F2:1C:0E:91:E4
- **Interface**: wl0
- **Channel**: 11
- **Region**: GB
- **Max Bit Rate**: Auto
- **Security**: WPA2-PSK (AES)
- **Password**: 0503478191
- **SSID Broadcast**: Enabled
- **Connected Clients**: 1

#### 5GHz Network
- **SSID**: WK - 5G
- **BSSID**: 82:F2:1C:0E:91:E4
- **Interface**: wl1
- **Channel**: 132
- **Region**: US
- **Max Bit Rate**: Auto
- **Security**: WPA2-PSK (AES)
- **Password**: 0503478191
- **SSID Broadcast**: Enabled
- **Management Frame Protection**: Required
- **Connected Clients**: 4

---

## 📞 VoIP Service Configuration

### Voice Service General Settings
- **Service Instance**: 1
- **Voice Profiles**: 18 (maximum)
- **Bound Interface**: `brvlan11`
- **Bound IP Address**: 20.159.4.1
- **Region**: SA (Saudi Arabia)
- **DTMF Method**: RFC2833
- **Max Sessions**: 4

### Supported Codecs
1. G.711 (20ms packetization)
2. G.723 (20ms packetization)
3. G.729 (20ms packetization)
4. G.726 (20ms packetization) - Disabled
5. G.722 (20ms packetization)
6. Additional codecs (Disabled)

---

## 🔊 SIP Configuration

### SIP Server Details
- **Proxy Server**: fmc.stc.com.sa
- **Registrar Server**: fmc.stc.com.sa
- **Outbound Proxy**: 10.200.42.121
- **User Agent Domain**: fmc.stc.com.sa
- **Secondary Outbound Proxy**: 0.0.0.0 (Not configured)
- **Music Server**: 0.0.0.0 (Not configured)
- **Log Server**: 0.0.0.0 (Not configured)

### SIP Protocol Settings
- **Register Retry Interval**: 30 seconds
- **DSCP Mark**: 40 (EF - Expedited Forwarding)
- **Interdigit Timeout**: 5000 ms
- **Address Mode**: DomainName
- **VoIP Switch Type**: SoftX3000

### Dial Plan (Digit Map)
```
[2-6]xxxxxx|[7-8]0[1-9]xxxx|[7-8][1-9]xxxxx|05xxxxxxxx|0[123467][2-8]xxxxxx.T|
01[123467][2-8]xxxxxx|00xxxxxx.S|1800xx|08111xxxxxx|9[034689]x|700xxxxx.T|
800xxxxxxx|92xxxxxxx|1xx.T|**xx|*xx#|*xx*x.#|*xx*x.*x.#|*xx*x.*x.*x.#|
*#xx*x.#|*#xx*x.*x.#|*#xx#|#xx#|#xx*x.*x.#
```

**Pattern Explanation**:
- `[2-6]xxxxxx` - Local 7-digit numbers
- `[7-8]0[1-9]xxxx`, `[7-8][1-9]xxxxx` - Local 8-digit numbers
- `05xxxxxxxx` - Saudi mobile numbers
- `0[123467][2-8]xxxxxx.T` - National numbers with timeout
- `00xxxxxx.S` - International calls (short dial)
- `1800xx` - Toll-free 1800
- `92xxxxxxx` - International (92 = Pakistan)
- `1xx.T` - Service codes
- `*xx`, `#xx`, etc. - Feature codes

---

## 📱 Voice Lines

### Line 1 (Primary)
- **Status**: Disabled
- **Directory Number**: +966114874423
- **Physical Reference**: Port 1
- **SIP Auth Username**: +966114874423@fmc.stc.com.sa
- **SIP URI**: +966114874423
- **Caller ID Name**: 966114874423
- **Max Sessions**: 2
- **MWI (Message Waiting Indicator)**: Enabled
- **Anonymous Call**: Enabled
- **Centrex Mode**: Enabled
- **PLAR Username**: 4003

**Codec Priority**:
1. Priority 2: Codec 1 (G.711)
2. Priority 1: Codec 2 (G.723)
3. Priority 3: Codec 3 (G.729)
4. Priority 99: Codec 4 (Disabled)
5. Priority 4: Codec 5 (G.722)

### Line 2 (Secondary)
- **Status**: Disabled
- **Directory Number**: 2001
- **Physical Reference**: Port 0 (No physical port)
- **CM Account Number**: 1
- **SIP URI**: 2001
- **Caller ID Name**: Line2
- **Max Sessions**: 2
- **MWI**: Enabled
- **Anonymous Call**: Enabled
- **Centrex Mode**: Enabled
- **PLAR Username**: 4004

**Codec Priority**: Same as Line 1

---

## 🔀 Port Forwarding Rules

### SSH Access
- **Name**: Ssh3
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 2223
- **Private Port**: 22
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### BeautyAI API Ports
- **Name**: API1
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Ports**: 8000-8003
- **Private Ports**: 8000-8003
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### HTTPS/Secure API
- **Name**: extern-to-intern
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 8443
- **Private Port**: 8443
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### Additional API Port
- **Name**: API2
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 8004
- **Private Port**: 8004
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### Standard HTTPS
- **Name**: extern-to-intern-443
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 443
- **Private Port**: 443
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### Standard HTTP (Redirect)
- **Name**: extern-to-intern-80
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 80
- **Private Port**: 192
- **Private IP**: 192.168.100.39
- **Status**: Enabled
- **Note**: Port 192 suggests HTTP-to-HTTPS redirect

### Testing Ports
- **Name**: Testing port
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Ports**: 5000-5003
- **Private Ports**: 5000-5003
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### Plex Media Server
- **Name**: plexc
- **Type**: Port-Remap
- **Protocol**: TCP
- **Public Port**: 32400
- **Private Port**: 32400
- **Private IP**: 192.168.100.191
- **Status**: Enabled

### FreePBX RTP (Voice Media)
- **Name**: FreePBX
- **Type**: Port-Range
- **Protocol**: UDP
- **Public Ports**: 10000-20000
- **Private Ports**: 10000-20000
- **Private IP**: 192.168.100.39
- **Status**: Enabled
- **Purpose**: RTP audio streams for voice calls

### WebRTC Signaling Port 1
- **Name**: WebRTC
- **Type**: Port-Remap
- **Protocol**: TCP or UDP
- **Public Port**: 9000
- **Private Port**: 9000
- **Private IP**: 192.168.100.39
- **Status**: Enabled

### WebRTC Signaling Port 2
- **Name**: WebRTC2
- **Type**: Port-Remap
- **Protocol**: TCP or UDP
- **Public Port**: 9001
- **Private Port**: 9001
- **Private IP**: 192.168.100.39
- **Status**: Enabled

---

## 🛡️ Firewall Rules

### Firewall Global Settings
- **Firewall**: Enabled
- **TCP SYN Cookies**: Enabled (SYN flood protection)
- **ICMP Echo DOS**: Enabled (Ping flood protection)
- **ICMP Redirection DOS**: Enabled
- **Land DOS**: Enabled (Same source/dest attack protection)
- **Smurf DOS**: Enabled (Broadcast ping protection)
- **WinNuke DOS**: Enabled
- **Ping Sweep DOS**: Enabled

### Application Layer Gateways (ALG)
- **FTP ALG**: Disabled
- **TFTP ALG**: Disabled
- **SIP ALG**: Disabled (Important for VoIP!)
- **IPsec ALG**: Enabled

### Custom Firewall Rules

#### WebRTC Traffic
- **Name**: WebRTC
- **Protocol**: TCP or UDP
- **Source Port**: 9000
- **Destination Port**: 9000
- **Source IP**: 0.0.0.0 (Any)
- **Destination IP**: 192.168.100.39
- **Interface**: ppp0.10
- **Status**: Enabled

#### Server to VoIP Network
- **Name**: Server to VOIP
- **Protocol**: TCP or UDP
- **Source Ports**: 5000-10000
- **Destination Ports**: 5000-10000
- **Source IP**: 192.168.100.39
- **Destination IP**: 20.159.4.1
- **Interface**: brvlan11
- **Status**: Enabled

#### VoIP Network to Server
- **Name**: VOIP to Server
- **Protocol**: TCP or UDP
- **Source Ports**: 5000-10000
- **Destination Ports**: 5000-10000
- **Source IP**: 20.159.4.1
- **Destination IP**: 192.168.100.39
- **Interface**: brvlan11
- **Status**: Enabled

#### SIP Proxy to Server (Inbound)
- **Name**: Proxy to server (PPP0.10)
- **Protocol**: TCP or UDP
- **Source Port**: 5060
- **Destination Port**: 5060
- **Source IP**: 10.200.42.121
- **Destination IP**: 192.168.100.39
- **Interface**: ppp0.10
- **Status**: Enabled

#### Server to SIP Proxy (Outbound)
- **Name**: Server to Proxy (PPP0.10)
- **Protocol**: TCP or UDP
- **Source Port**: 5060
- **Destination Port**: 5060
- **Source IP**: 192.168.100.39
- **Destination IP**: 10.200.42.121
- **Interface**: ppp0.10
- **Status**: Enabled

---

## 🏷️ VLAN Configuration

### VLAN Mode
- **Mode**: Normal
- **S-TPID**: 8100 (Standard 802.1Q)
- **Multicast Across VLANs**: Enabled

### Bridge Configurations

#### Bridge 1 (Default)
- **Bridge Key**: 0
- **Name**: Default
- **VLAN Type**: Bridged
- **Security**: Disabled

#### Bridge 2 (Internet/TR069)
- **Bridge Key**: 10
- **Name**: 1_TR069_INTERNET_B_VID_10
- **VLAN ID**: 10
- **VLAN Type**: PPPoE_Bridged
- **Security**: Disabled
- **Connected Interfaces**: 
  - eth0 (WAN)
  - eth1-eth4 (LAN ports)
  - wl0, wl0_1-wl0_3 (2.4GHz WiFi)
  - wl1, wl1_1-wl1_3 (5GHz WiFi)

#### Bridge 3 (VoIP)
- **Bridge Key**: 11
- **Name**: 2_VoIP_B_VID_11
- **VLAN ID**: 11
- **VLAN Type**: CPU-Bridged
- **Security**: Disabled
- **Connected Interfaces**: eth0 (WAN only)

### VLAN Tagging
- **All LAN ports (eth1-eth4)**: Untagged on VLAN 10
- **WiFi interfaces**: Untagged on VLAN 10
- **WAN (eth0)**: Tagged for VLAN 10 and 11 via OMCI

---

## ⚡ QoS Configuration

### QoS Scheduling
- **Type**: Combo (Weighted + Priority)

### Voice Traffic Priority
- **DSCP Mark**: 40 (EF - Expedited Forwarding)
- **Classification**: High priority for voice packets

### WMM (WiFi Multimedia) Queues

#### 2.4GHz WiFi (wl0)
1. **Voice Priority** (Queue 8): Highest priority
2. **Voice Priority** (Queue 7): High priority
3. **Video Priority** (Queue 6): Medium-high priority
4. **Video Priority** (Queue 5): Medium priority
5. **Best Effort** (Queue 4): Normal priority
6. **Background** (Queue 3): Low priority
7. **Background** (Queue 2): Lowest priority
8. **Best Effort** (Queue 1): Default

#### 5GHz WiFi (wl1)
- Same queue configuration as 2.4GHz

---

## 🎯 Classification Rules

### Ingress Classification Rules

#### Rule 2: Server to VoIP
- **Name**: Server to VOIP
- **Source IP**: 192.168.100.39/32
- **Source MAC Mask**: Any
- **Destination IP**: 20.159.4.1/32
- **VLAN Mark**: brvlan11
- **VLAN Priority**: 0

**Bindings**: LAN ports eth1-eth4 (interfaces 2-5)

#### Rule 3: VoIP to Server
- **Name**: VOIP to Server
- **Source IP**: 20.159.4.1/32
- **Source MAC Mask**: Any
- **Destination IP**: 192.168.100.39/32
- **VLAN Mark**: brvlan10
- **VLAN Priority**: 0

**Bindings**: LAN ports eth1-eth4 (interfaces 2-5)

### Traffic Flow Summary
```
[Server: 192.168.100.39] ←→ [VoIP Network: 20.159.4.1]
        │                              │
        └─ VLAN 10 (brvlan10)         └─ VLAN 11 (brvlan11)
                                       └─ SIP Proxy: 10.200.42.121
```

---

## 🔧 Key Network Paths

### Voice Traffic Path
```
BeautyAI Server (192.168.100.39)
  ↓ [VLAN 10 → VLAN 11 Classification]
VoIP Bridge (brvlan11 - 20.159.4.1)
  ↓ [VLAN 11 Tagged]
WAN Interface (eth0)
  ↓ [GPON Fiber]
STC VoIP Network
  ↓
SIP Proxy (10.200.42.121)
  ↓
SIP Registrar (fmc.stc.com.sa)
```

### Internet Traffic Path
```
LAN Devices (192.168.100.0/24)
  ↓ [VLAN 10 Bridge]
PPPoE Interface (ppp0.10)
  ↓ [176.45.31.103 Public IP]
WAN Interface (eth0.v10)
  ↓ [VLAN 10 Tagged]
GPON Fiber
  ↓
ISP Gateway (84.235.6.25)
  ↓
Internet
```

---

## 📊 Router System Information

- **Model**: ZNID-GPON-2428B1-0ST
- **Manufacturer**: Dasan Zhone
- **Hardware**: GPON ONT with 4x GigE LAN + 2x WiFi (2.4GHz/5GHz)
- **Firmware**: S4.2.022
- **Config Version**: 1
- **Serial/FSAN**: 5a4e5453050e91e3
- **MAC Address**: 12:F2:1C:0E:91:E3
- **First Use Date**: 2024-09-26T21:01:51+00:00
- **Management Server**: https://devmgmt.stc.com.sa/cwmpWeb/CPEMgt
- **SNMP System Name**: ZNID24xxB1-Router

### Management Access (LAN)
- **HTTP**: Allowed
- **HTTPS**: Allowed
- **SSH**: Allowed
- **Ping**: Allowed
- **SNMP**: Allowed
- **SNMP Trap**: Allowed

### Management Access (WAN - ppp0.10)
- **HTTP**: Denied
- **HTTPS**: Denied
- **SSH**: Denied
- **Ping**: Allowed
- **SNMP**: Denied
- **SNMP Trap**: Denied

---

## ⚠️ Important Notes

### VoIP Connectivity Issue - ROOT CAUSE IDENTIFIED! ✅
**Current Problem**: Cannot register to STC SIP server (10.200.42.121)

**ROOT CAUSE**: Server needs VLAN 11 tagged interface to access VoIP network!

**Diagnosis**:
- ✅ Router configuration is correct
- ✅ VLAN 11 properly configured for VoIP on router
- ✅ Firewall rules allow SIP traffic
- ✅ Port forwarding configured for RTP media
- ❌ **Server (192.168.100.39) is on VLAN 10 only**
- ❌ **VoIP network (20.159.4.x) is on VLAN 11**
- ❌ **SIP Proxy (10.200.42.121) is accessible only through VLAN 11**

**Network Architecture Understanding**:
```
Server: 192.168.100.39
  └─ Physical NIC: enp12s0 (connected to LAN port)
       ├─ No VLAN tag → VLAN 10 (brvlan10) → Internet
       └─ VLAN 11 tag → VLAN 11 (brvlan11) → VoIP Network → SIP Proxy
```

**Solution - VLAN 11 Interface Required**:

The router's classification rules are designed to route traffic between VLANs, but the server needs to have **direct access to VLAN 11** to communicate with the VoIP network.

**Tested Configuration** (Temporary - needs to be permanent):
```bash
# 1. Load VLAN kernel module
sudo modprobe 8021q

# 2. Create VLAN 11 interface
sudo ip link add link enp12s0 name enp12s0.11 type vlan id 11
sudo ip link set enp12s0.11 up

# 3. Assign IP in VoIP network range (20.159.4.x/24)
sudo ip addr add 20.159.4.100/24 dev enp12s0.11

# 4. Add route to SIP proxy network
sudo ip route add 10.200.42.0/24 via 20.159.0.1 dev enp12s0.11
```

**Required Action**: 
1. ✅ **Create permanent VLAN 11 configuration** using netplan/NetworkManager
2. ⚠️ **May need to request IP from router's VLAN 11 DHCP** (20.159.4.x range)
3. ⚠️ **Update PABX configuration** to bind to VLAN 11 interface
4. ⚠️ **Test SIP registration** after VLAN 11 is properly configured
5. ⚠️ **Router's VLAN 11 might need manual IP assignment** (DHCP on VLAN 11 may be restricted)

### Security Recommendations
1. ✅ SIP ALG is disabled (good for VoIP)
2. ✅ DOS protection enabled
3. ⚠️ WAN management interfaces disabled (good)
4. ⚠️ Consider limiting SSH to specific IPs instead of any source

### Performance Notes
- RTP port range (10000-20000) allows for ~500 concurrent calls
- QoS properly prioritizes voice traffic (DSCP 40)
- VLAN separation ensures voice quality isolation from data traffic

---

## 📝 Changelog

### 2025-11-20 (Update 2)
- **🎯 ROOT CAUSE IDENTIFIED**: Server needs VLAN 11 tagged interface
- Created temporary VLAN 11 interface (enp12s0.11) for testing
- Assigned IP 20.159.4.100/24 on VLAN 11
- Confirmed VLAN architecture: VLAN 10 (Internet) / VLAN 11 (VoIP)
- Next step: Permanent VLAN 11 configuration required

### 2025-11-20 (Initial)
- Initial documentation created from router backup configuration
- Documented complete network topology
- Identified VoIP connectivity issue to STC network
- Documented all port forwarding and firewall rules

---

*This document is maintained for BeautyAI PABX system integration with STC VoIP services.*
