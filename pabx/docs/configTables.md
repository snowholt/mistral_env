
## PPPOE Interface Setup

### Interface 1: eth0.v10.ppp

#### Interface Attributes
*   **I/F Name:** Fiber WAN
*   **I/F Type:** Uplink
*   **VLAN ID:** 10
*   **MAC Address:** `40:f2:1c:0e:91:e3`

#### IPv4 Configuration
*   **Address Mode:** PPPoE
*   **IP address:** `176.45.31.103`
*   **Subnet Mask:** `255.255.255.255`
*   **Default Gateway:** `84.235.6.25`
*   **Max MTU Size:** 1492

#### IPv4 Client Addressing
*   **NAT/NAPT:** NAT
*   **UPnP:** Disable
*   **DNS Primary:** `84.235.6.55`
*   **DNS Secondary:** `84.235.57.230`

#### IPv6 Configuration
*   **Address Mode:** Unnumbered_DHCP-PD
*   **DUID:** `00:03:00:01:40:f2:1c:0e:91:e3`
*   **Public address:** `::/128`
*   **Local address:** `fe80::42f2:1cff:fe0e:91e3/10`
*   **Default Gateway:** `fe80::7a19:f7ff:fe45:28f4`
*   **DNS Server Address:** `::/128`

#### IPv6 Client Addressing
*   **Delegated Prefix:** `2001:16a2:dfe9:5d00::/56`
*   **DNS Primary:** `::/128`


### Interface 2: brvlan10

#### Interface Attributes
*   **I/F Name:** Bridge
*   **VLAN ID:** 10

#### IPv4 Configuration
*   **Address Mode:** Static
*   **IP address:** `192.168.100.1`
*   **Subnet Mask:** `255.255.255.0`
*   **Max MTU Size:** 1492

#### IPv4 Client Addressing
*   **DHCP Server:** Enabled
*   **Normal Range:** `192.168.100.2 - 192.168.100.254`
*   **Conditional DHCP:** Disabled
*   **Lease Duration (sec):** 86400
*   **DNS Relay Source:** Default
*   **DNS Secondary:** `84.235.57.230`

#### IPv6 Configuration
*   **Address Mode:** LAN-PD
*   **Public address:** `2001:16a2:dfe9:5d00:10f2:1cff:fe0e:91e3/64`
*   **Local address:** `fe80::10f2:1cff:fe0e:91e3/64`

#### IPv6 Client Addressing
*   **DHCPv6 Server:** Stateless
*   **Router Advertisement:** Enabled
*   **DNS Relay Source:** Default

---



## PPPOE Interface Setup

### Interface 1: eth0.v10.ppp

#### Interface Attributes
*   **I/F Name:** Fiber WAN
*   **I/F Type:** Uplink
*   **VLAN ID:** 10
*   **MAC Address:** `40:f2:1c:0e:91:e3`

#### IPv4 Configuration
*   **Address Mode:** PPPoE
*   **IP address:** `176.45.31.103`
*   **Subnet Mask:** `255.255.255.255`
*   **Default Gateway:** `84.235.6.25`
*   **Max MTU Size:** 1492

#### IPv4 Client Addressing
*   **NAT/NAPT:** NAT
*   **UPnP:** Disable
*   **DNS Primary:** `84.235.6.55`
*   **DNS Secondary:** `84.235.57.230`

#### IPv6 Configuration
*   **Address Mode:** Unnumbered_DHCP-PD
*   **DUID:** `00:03:00:01:40:f2:1c:0e:91:e3`
*   **Public address:** `::/128`
*   **Local address:** `fe80::42f2:1cff:fe0e:91e3/10`
*   **Default Gateway:** `fe80::7a19:f7ff:fe45:28f4`
*   **DNS Server Address:** `::/128`

#### IPv6 Client Addressing
*   **Delegated Prefix:** `2001:16a2:dfe9:5d00::/56`
*   **DNS Primary:** `::/128`


### Interface 2: brvlan10

#### Interface Attributes
*   **I/F Name:** Bridge
*   **VLAN ID:** 10

#### IPv4 Configuration
*   **Address Mode:** Static
*   **IP address:** `192.168.100.1`
*   **Subnet Mask:** `255.255.255.0`
*   **Max MTU Size:** 1492

#### IPv4 Client Addressing
*   **DHCP Server:** Enabled
*   **Normal Range:** `192.168.100.2 - 192.168.100.254`
*   **Conditional DHCP:** Disabled
*   **Lease Duration (sec):** 86400
*   **DNS Relay Source:** Default
*   **DNS Secondary:** `84.235.57.230`

#### IPv6 Configuration
*   **Address Mode:** LAN-PD
*   **Public address:** `2001:16a2:dfe9:5d00:10f2:1cff:fe0e:91e3/64`
*   **Local address:** `fe80::10f2:1cff:fe0e:91e3/64`

#### IPv6 Client Addressing
*   **DHCPv6 Server:** Stateless
*   **Router Advertisement:** Enabled
*   **DNS Relay Source:** Default

---


## SIP Proxy & Registrar Settings

*   **SIP Proxy:** `fmc.stc.com.sa`
*   **SIP Proxy Port:** `5060`
*   **SIP Outbound Proxy:** `10.200.42.121`
*   **SIP Outbound Proxy Port:** `5060`
*   **SIP Secondary Outbound Proxy:** `0.0.0.0`
*   **SIP Registrar:** `fmc.stc.com.sa`
*   **SIP Registrar Port:** `5060`

### General & Network Settings

*   **Bound Interface Name:** `brvlan11 => 20.159.4.1`
*   **Locale Selection:** `KSA`
*   **Domain Name Mode:** `DomainName`
*   **SIP Domain Name:** `fmc.stc.com.sa`
*   **SIP Transport Protocol:** `UDP`

### Advanced Settings

*   **Registration Expire Timeout:** `3600`
*   **Registration Retry Interval:** `30`
*   **Head Start Value (secs):** `15`
*   **Enable T38 support:** Disabled (Value is "1")
*   **WAN MAC + 1:** Enabled (Value is "1")
*   **DSCP for SIP:** `40`
*   **DSCP for RTP:** `46`
*   **DTMF Relay Setting:** `RFC2833`
*   **Hook Flash Relay Setting:** `None`
*   **MWI Subscribe:** `Enable` (Value is "1")
*   **Switch Model:** `SoftX3000`
*   **InterDigit Delay:** `5`
*   **Conference URI:** (empty)
*   **Conference Option:** `Local`
*   **Voip Dial Plan Setting:** `[2-6]xxxxxx|[7-8]0[1-9]xxxx|[7-8][1-9]xxxxx|05xxxxxxxx|0[123467][2-8]xxxxxx.T|01[123467][2-8]xxxxxx|00xxxxxx.S|1800xx|08111xxxxxx|9[034689]x|700xxxxx.T|800xxxxxxx|92xxxxxxx|1xx.T|**xx|*xx#|*xx*x.#|*xx*x.*x.#|*xx*x.*x.*x.#|*#xx*x.#|*#xx*x.*x.#|*#xx#|#xx#|#xx*x.*x.#`



---

## Bridge Interface Setup

### Bridge Interface: brvlan11

#### Interface Attributes
*   **VLAN Name:** `2_VoIP_B_VID_11`
*   **VLAN ID:** `11`
*   **IGMP Snooping:** `Disabled`
*   **IGMP Proxy:** `Disabled`
*   **IGMP Forking:** `Disabled`
*   **IGMP Version 3:** `Disabled`
*   **IGMP Querier:** `Disabled`

#### IP Configuration
*   **Address Mode:** `DHCP`
*   **IP address:** `20.159.4.1`
*   **Subnet Mask:** `255.255.224.0`
*   **Default Gateway:** `20.159.0.1`
*   **Max MTU Size:** `1500`
*   **Cross-Routing NAT:** `NAT`


---

## Transparent LAN Service Settings

*   **VLAN Service Mode:** `Normal`
*   **Bridge Loop Detect:** `Enable`
*   **Block Upstream DHCP Offer:** `Enable`
*   **Block Downstream DNS Query:** `Enable`
*   **Local Switching Method:** `Auto` (Current running method is `FlowCache`)

### Global VLAN Routing Mode
*   **Cross VLAN Routing:** `Enable`
*   **Cross VLAN Multicast:** `Enable`

### Global QOS Mode
*   **Prioritization Method:** `Layer2 VLAN CoS`
*   **Scheduling Method:** `Strict with Weighted Round Robin`



---


## VLAN Filter Rules Setup

### Rule 1: Server to VOIP

#### Rule Details
*   **Ingress Filter Rule Name:** `Server to VOIP`
*   **Administration State:** `Enable`
*   **Rule Priority:** `1`

#### Ingress Classification
*   **VLAN Traffic:** `untagged`
*   **Source MAC:** `Any`
*   **MAC Mask:** `Any`
*   **Ethernet Type:** `Any`
*   **Source IP Address:** `192.168.100.39/32`
*   **Destination IP Address:** `20.159.4.1/32`
*   **DSCP:** `Any`

#### Action
*   **Discard All Packets:** `No`
*   **VLAN Service:** `Normal`
*   **VLAN TPID:** `8100`
*   **VLAN ID:** `brvlan11`
*   **VLAN 802.1p:** `0`
*   **DSCP:** `None`

#### Ingress Ports
*   **GE1 - GigE:** `Y`
*   **GE2 - GigE:** `Y`
*   **GE3 - GigE:** `Y`
*   **GE4 - GigE:** `Y`


### Rule 2: VOIP to Server

#### Rule Details
*   **Ingress Filter Rule Name:** `VOIP to Server`
*   **Administration State:** `Enable`
*   **Rule Priority:** `1`

#### Ingress Classification
*   **VLAN Traffic:** `untagged`
*   **Source MAC:** `Any`
*   **MAC Mask:** `Any`
*   **Ethernet Type:** `Any`
*   **Source IP Address:** `20.159.4.1/32`
*   **Destination IP Address:** `192.168.100.39/32`
*   **DSCP:** `Any`

#### Action
*   **Discard All Packets:** `No`
*   **VLAN Service:** `Normal`
*   **VLAN TPID:** `8100`
*   **VLAN ID:** `brvlan10`
*   **VLAN 802.1p:** `0`
*   **DSCP:** `None`

#### Ingress Ports
*   **GE1 - GigE:** `Y`
*   **GE2 - GigE:** `Y`
*   **GE3 - GigE:** `Y`
*   **GE4 - GigE:** `Y`

---

## VLAN Network Setup:

### Port Defaults

| Port Name | Uplink | Default PVID | Default 802.1p | IGMP PVID | IGMP 802.1p |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fiber WAN (eth0)** | Uplink | 10 | 0 | 0 | 0 |
| **GE1 - GigE (eth1)** | Uplink | 10 | 0 | 0 | 0 |
| **GE2 - GigE (eth2)** | - | 10 | 0 | 0 | 0 |
| **GE3 - GigE (eth3)** | - | 10 | 0 | 0 | 0 |
| **GE4 - GigE (eth4)** | - | 10 | 0 | 0 | 0 |
| **2.4G - SSID 0 (wl0)** | - | 10 | 0 | 0 | 0 |
| **2.4G - SSID 1 (wl0_1)** | - | 10 | 0 | 0 | 0 |
| **2.4G - SSID 2 (wl0_2)** | - | 10 | 0 | 0 | 0 |
| **2.4G - SSID 3 (wl0_3)** | - | 10 | 0 | 0 | 0 |
| **5G - SSID 0 (wl1)** | - | 10 | 0 | 0 | 0 |
| **5G - SSID 1 (wl1_1)** | - | 10 | 0 | 0 | 0 |
| **5G - SSID 2 (wl1_2)** | - | 10 | 0 | 0 | 0 |
| **5G - SSID 3 (wl1_3)** | - | 10 | 0 | 0 | 0 |



### VLAN Configuration

#### VLAN 10
*   **VLAN ID:** `10`
*   **VLAN Name:** `1_TR069_INTERNET_B_VID_10`
*   **Connection Type:** `PPPoE_Bridged`
*   **Secure Forwarding:** `Disable`
*   **Port Membership:**
    *   Fiber WAN (eth0): `O`
    *   GE1 - GigE (eth1): `U`
    *   GE2 - GigE (eth2): `U`
    *   GE3 - GigE (eth3): `U`
    *   GE4 - GigE (eth4): `U`
    *   2.4G - SSID 0 (wl0): `U`
    *   5G - SSID 0 (wl1): `U`
    *   5G - SSID 1 (wl1_1): `U`

#### VLAN 11
*   **VLAN ID:** `11`
*   **VLAN Name:** `2_VoIP_B_VID_11`
*   **Connection Type:** `CPU-Bridged`
*   **Secure Forwarding:** `Disable`
*   **Port Membership:**
    *   Fiber WAN (eth0): `O`



---
