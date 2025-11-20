### Logs
- HT813 log: `http://192.168.100.39:8080/api/syslog/messages`
- Pabx log: `pabx/logs/system/journal_pabx_backend.log`

### Task
- We checked the PSTN line with provider, an they provided these info to me: 

This is the correct information for Voice SIP Line Configuration:
SIP Outbound Proxy: 10.200.42.121
User ID: +966114874423
Authentication name: +966114874423@fmc.stc.com.sa
Password: PaSsWoRd


This passord is the encrypted password


The real password is 
114874423114874423

Tested and confirmed ✅


We need to ensure that our communication from our server to proxy is active


VoIP IP:
20.159.4.1



-----
The router settings: 
Based on the image provided, here is the extracted information from the Voice/SIP configuration page:

| Setting | Value |
| :--- | :--- |
| **Locale selection** | KSA - SAUDIARABIA |
| **Domain Name Mode** | Domain Name |
| **SIP domain name** | fmc.stc.com.sa |
| **SIP Usage** | Use SIP (checked) |
| **SIP Proxy** | fmc.stc.com.sa |
| **SIP Proxy port** | 5060 |
| **SIP Outbound Proxy** | 10.200.42.121 |
| **SIP Secondary Outbound Proxy** | 0.0.0.0 |
| **SIP Outbound Proxy port** | 5060 |
| **SIP Registrar** | fmc.stc.com.sa |
| **SIP Registrar port** | 5060 |
| **Enable T38 support** | Unchecked |
| **WAN MAC + 1** | Checked |
| **Registration Expire Timeout** | 3600 |
| **Head Start Value (secs)** | 15 |
| **Registration Retry Interval** | 30 |
| **Voip Dial Plan Setting** | [2-6]xxxxxx|[7-8]0|[1-9] |
| **DSCP for SIP** | 40 (40 - CS5 (101000)) |
| **DSCP for RTP** | 46 (46 - EF (101110)) |
| **Dtmf Relay setting** | RFC2833 |
| **Hook Flash Relay setting** | None |
| **SIP Transport protocol** | UDP |
| **MWI Subscribe** | Enable |
| **Browser Address Bar** | 192.168.100.1 |



---
## HT813 Settings Extraction
### Basic Settings:
Based on the provided HTML for the Grandstream HT813's "BASIC SETTINGS" page, here is the extracted information, including current values and selected options:

## Grandstream HT813 Basic Settings

### Password Configuration

| Setting | Value/Note |
| :--- | :--- |
| **New End User Password** | (Input field, purposely not displayed) |
| **Confirm End User Password** | (Input field, purposely not displayed) |
| **New Viewer Password** | (Input field, purposely not displayed) |
| **Confirm Viewer Password** | (Input field, purposely not displayed) |

### Web/SSH Access

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *Web Session Timeout* | 60 | (1-60, default 10 minutes) |
| *Web Access Attempt Limit* | 5 | (1-10, default 5) |
| *Web Lockout Duration* | 15 | (0-60, default 15 minutes) |
| *Web Access Mode* | HTTP (checked) | HTTPS / HTTP |
| *HTTP Web Port* | 80 | (default 80) |
| *HTTPS Web Port* | 443 | (default 443) |
| *Disable SSH* | No (checked) | No / Yes |
| *SSH Port* | 22 | (default 22) |
| *Disable Telnet* | Yes (checked) | No / Yes |
| *Telnet Port* | 23 | (default 23) |
| *WAN Side Web/SSH Access* | Auto (checked) | No / Yes / Auto |
| *White List for WAN Side* | (Empty Textarea) | |
| *Black List for WAN Side* | (Empty Textarea) | |

### Internet Protocol

| Setting | Value | Options |
| :--- | :--- | :--- |
| **Internet Protocol** | IPv4 Only (checked) | IPv4 Only / IPv6 Only / Both, prefer IPv4 / Both, prefer IPv6 |

#### IPv4 Address Configuration

| Setting | Value |
| :--- | :--- |
| **IPv4 Address Mode** | dynamically assigned via DHCP (checked) |
| *DHCP hostname* | (Empty) |
| *DHCP domain name* | (Empty) |
| *DHCP vendor class ID* | HT8XX |
| **PPPoE Account ID** | (Empty) |
| **PPPoE Password** | (Empty) |
| **PPPoE Service Name** | (Empty) |
| **1st Preferred DNS server** | 0.0.0.0 |
| **2nd Preferred DNS server** | 0.0.0.0 |
| **3rd Preferred DNS server** | 0.0.0.0 |
| **4th Preferred DNS server** | 0.0.0.0 |
| **Static IP Address** | 192.168.0.160 (if static mode selected) |
| **Subnet Mask** | 255.255.0.0 (if static mode selected) |
| **Default Router** | 0.0.0.0 (if static mode selected) |
| **DNS Server 1** | 0.0.0.0 (if static mode selected) |
| **DNS Server 2** | 0.0.0.0 (if static mode selected) |

#### IPv6 Address Configuration

| Setting | Value |
| :--- | :--- |
| **IPv6 Address Mode** | dynamically assigned via DHCP (checked) |
| *Static Mode Type* | Full Static (checked, if static mode selected) |
| *Static IPv6 Address* | (Empty) |
| *IPv6 Prefix Length* | (Empty) |
| *Prefix Static IPv6 Prefix* | (Empty) |
| *DNS Server 1* | (Empty) |
| *DNS Server 2* | (Empty) |
| *Preferred DNS Server* | (Empty) |

### Time and Language

| Setting | Value |
| :--- | :--- |
| **Time Zone** | Using self-defined Time Zone (selected) |
| *Self-Defined Time Zone* | MTZ+6MDT+5,M3.2.0,M11.1.0 |
| *Allow DHCP server to set Time Zone* | Yes (checked) |
| **Language** | English (selected) |

### NAT/DHCP Server Information & Configuration

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *Device Mode* | Bridge (checked) | NAT Router / Bridge / WAN Only |
| *NAT maximum ports* | 1024 | (0 - 4096, default 1024) |
| *NAT TCP timeout* | 3600 | (0 - 3600, default 3600) |
| *NAT UDP timeout* | 300 | (0 - 3600, default 300) |
| *Uplink bandwidth* | Disabled (selected) | |
| *Downlink bandwidth* | Disabled (selected) | |
| *Enable UPnP support* | No (checked) | No / Yes |
| *Reply to ICMP on WAN port* | Yes (checked) | No / Yes |
| *Cloned WAN MAC Addr* | (Empty fields) | |
| *LAN Port VLAN Tag* | 0 | (0-4094) |
| *LAN Port Priority Value* | 0 | (0-7) |
| *Enable LAN DHCP* | Yes (checked) | No / Yes |
| *LAN DHCP Base IP* | 192.168.2.1 | (default 192.168.2.1) |
| *LAN DHCP Start IP* | 192.168.2.100 | (default 192.168.2.100) |
| *LAN DHCP End IP* | 192.168.2.199 | (default 192.168.2.199) |
| *LAN Subnet Mask* | 255.255.255.0 | (default 255.255.255.0) |
| *DHCP IP Lease Time* | 120 | (in hours, default 120) |
| *DMZ IP* | (Empty) | |

### Port Forwarding (8 Entries)

| Entry | WAN Port | LAN IP | LAN Port | Protocol |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0 | (Empty) | 0 | UDP Only (selected) |
| 2 | 0 | (Empty) | 0 | UDP Only (selected) |
| 3 | 0 | (Empty) | 0 | UDP Only (selected) |
| 4 | 0 | (Empty) | 0 | UDP Only (selected) |
| 5 | 0 | (Empty) | 0 | UDP Only (selected) |
| 6 | 0 | (Empty) | 0 | UDP Only (selected) |
| 7 | 0 | (Empty) | 0 | UDP Only (selected) |
| 8 | 0 | (Empty) | 0 | UDP Only (selected) |

### Reset Configuration

| Setting | Value |
| :--- | :--- |
| **Reset Type** | ISP Data Reset (selected) |

### PSTN/VoIP Call Forwarding

| Setting | Value |
| :--- | :--- |
| *PSTN Access Code* | *00 |
| *PIN for VoIP-to-PSTN Calls* | (Empty) |
| *PIN for PSTN-to-VoIP Calls* | (Empty) |
| *Unconditional Call Forward to PSTN* | (Empty) |
| *Unconditional Call Forward to VOIP* | 1001@192.168.100.39:5060 |



### Advanced Settings:
Based on the provided HTML for the Grandstream HT813's "ADVANCED SETTINGS" page, here is the extracted configuration information:

## Grandstream HT813 Advanced Settings

### Administrative Access

| Setting | Value/Note |
| :--- | :--- |
| **New Admin Password** | (Input field, purposely not displayed) |
| **Confirm Admin Password** | (Input field, purposely not displayed) |
| **Disable User Level Web Access** | Yes (checked) |
| **Disable Viewer Level Web Access** | Yes (checked) |

### Network and QoS

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *802.1Q/VLAN Tag* | 0 | (0-4094) |
| *SIP 802.1p* | 0 | (0-7) |
| *RTP 802.1p* | 0 | (0-7) |
| *Black List for WAN Side Port* | (Empty Textarea) | |
| *STUN server is* | (Empty) | (URI or IP:port) |
| *Keep-alive Interval* | 20 | (in seconds, default 20) |
| *Use STUN to detect network connectivity* | No (checked) |
| *STUN response misses to restart DHCP* | 3 | (minimum 3) |
| *Use DNS to detect network connectivity* | No (checked) |
| *Verify host when using HTTPS* | No (checked) |

### Firmware Upgrade and Provisioning

| Setting | Value | Options/Note |
| :--- | :--- | :--- |
| *Upgrade Via* | HTTPS (checked) | TFTP / HTTP / HTTPS / FTP / FTPS |
| *Firmware Server Path* | fm.grandstream.com/gs | |
| *Config Server Path* | fm.grandstream.com/gs | |
| *XML Config File Password* | (Empty Password Field) | |
| *HTTP/HTTPS/FTP/FTPS User Name* | (Empty) | |
| *HTTP/HTTPS/FTP/FTPS Password* | (Empty Password Field) | |
| *Firmware File Prefix* | (Empty) | |
| *Firmware File Postfix* | (Empty) | |
| *Config File Prefix* | (Empty) | |
| *Config File Postfix* | (Empty) | |
| *Enable using tags in URL* | No (checked) | |
| *Always send HTTP Basic Authentication Information* | No (checked) | |
| *Allow DHCP Option 66 or 160 to override server* | Yes (checked) | |
| *Additional Override DHCP Option* | None (selected) | Option 150 |
| *3CX Auto Provision* | Yes (checked) | |
| *Automatic Upgrade* | No (checked) |
| *Automatic Upgrade Interval* | 10080 minutes | (if enabled, 30-5256000) |
| *Automatic Upgrade Daily Start Hour* | 1 | (0-23) |
| *Automatic Upgrade Daily End Hour* | 22 | (0-23) |
| *Automatic Upgrade Weekly Day* | 1 | (0-6) |
| *Randomized Automatic Upgrade* | No (checked) | |
| *Firmware Check at Boot up* | Always Check for New Firmware at Boot up (checked) |
| *Configuration File Types Allowed* | All (selected) | XML only |
| *Download and Process All Available Config Files* | No (checked) | |
| *Disable SIP NOTIFY Authentication* | No (checked) | |
| *Authenticate Conf File* | No (checked) | |
| *Validate Server Certificates* | No (checked) | |
| *Trusted CA Certificates A* | (Empty Textarea) | |
| *Trusted CA Certificates B* | (Empty Textarea) | |
| *Load CA Certificates* | Built-in trusted certificates (selected) | Custom / All trusted certificates |
| *SIP TLS Certificate* | (Empty Textarea) | |
| *SIP TLS Private Key* | (Empty Textarea) | |
| *SIP TLS Private Key Password* | (Empty Password Field) | |
| *Custom Certificate* | (Empty Textarea) | |

### TR-069 (ACS)

| Setting | Value |
| :--- | :--- |
| *Enable TR-069* | Yes (checked) |
| *ACS URL* | https://acsguestb.gdms.cloud |
| *ACS Username* | (Empty) |
| *ACS Password* | (Empty Password Field) |
| *Periodic Inform Enable* | Yes (checked) |
| *Periodic Inform Interval* | 86400 |
| *Connection Request Username* | EC74D74LDC4A |
| *Connection Request Password* | (Empty Password Field) |
| *Connection Request Port* | 7547 |
| *CPE SSL Certificate* | (Empty Textarea) |
| *CPE SSL Private Key* | (Empty Textarea) |

### SNMP

| Setting | Value |
| :--- | :--- |
| *Enable SNMP* | No (checked) |
| *SNMP Version* | Version 3 (selected) |
| *SNMP Port* | 161 |
| *SNMP Trap IP Address* | (Empty) |
| *SNMP Trap Port* | 162 |
| *SNMP Trap Version* | Version 2c (selected) |
| *SNMP Trap Interval* | 5 |
| *SNMPv1/v2c Community* | (Empty) |
| *SNMPv1/v2c Trap Community* | (Empty) |
| *SNMPv3 User Name* | (Empty) |
| *SNMPv3 Security Level* | noAuthUser (selected) |
| *SNMPv3 Authentication Protocol* | None (selected) |
| *SNMPv3 Privacy Protocol* | None (selected) |
| *SNMPv3 Authentication Key* | (Empty Textarea) |
| *SNMPv3 Privacy Key* | (Empty Textarea) |
| *SNMPv3 Trap User Name* | (Empty) |
| *SNMPv3 Trap Security Level* | noAuthUser (selected) |
| *SNMPv3 Trap Authentication Protocol* | None (selected) |
| *SNMPv3 Trap Privacy Protocol* | None (selected) |
| *SNMPv3 Trap Authentication Key* | (Empty Textarea) |
| *SNMPv3 Trap Privacy Key* | (Empty Textarea) |

### RADIUS Web Access Control

| Setting | Value |
| :--- | :--- |
| *Enable RADIUS Web Access Control* | No (checked) |
| *Action upon Radius Auth Server Error* | Authenticate Locally (checked) |
| *RADIUS Auth Server Address* | (Empty) |
| *RADIUS Auth Server Port* | 1812 |
| *RADIUS Shared Secret* | (Empty Password Field) |
| *RADIUS VSA Vendor ID* | 42397 |
| *RADIUS VSA Access Level Attribute* | (Empty) |

### DDNS

| Setting | Value |
| :--- | :--- |
| *Enable DDNS* | No (checked) |
| *DDNS Server* | dyndns.org (selected) |
| *DDNS Username* | (Empty) |
| *DDNS Password* | (Empty Password Field) |
| *DDNS Hostname* | (Empty) |
| *DDNS Hash* | (Empty) |

### Tones and Prompts

| Setting | Value |
| :--- | :--- |
| *System Ring Cadence* | c=2000/4000; |
| *Dial Tone* | f1=350@-17,f2=440@-17,c=0/0; |
| *Ringback Tone* | f1=440@-17,f2=480@-17,c=2000/4000; |
| *Busy Tone* | f1=480@-21,f2=620@-21,c=500/500; |
| *Reorder Tone* | f1=480@-21,f2=620@-21,c=250/250; |
| *Confirmation Tone* | f1=350@-11,f2=440@-11,c=100/100-100/100-100/100; |
| *Call Waiting Tone* | f1=440@-13,c=300/10000; |
| *Prompt Tone* | f1=350@-17,f2=440@-17,c=0/0; |
| *Prompt Tone Access Code* | (Empty) |
| *Lock Keypad Update* | No (checked) |
| *Disable Voice Prompt* | No (checked) |
| *Disable Direct IP Call* | No (checked) |

### Miscellaneous

| Setting | Value |
| :--- | :--- |
| *Life Line Mode* | Auto (checked) |
| *Blacklist For Incoming Calls* | (Empty Textarea) |
| *NTP Server* | pool.ntp.org |
| *Allow DHCP Option 42 to override NTP server* | Yes (checked) |
| *DHCP Option 17 Enterprise Number* | 3561 |
| *Disable Weak TLS Cipher Suites* | Enable Weak TLS Ciphers Suites (selected) |
| *Minimum TLS Version* | Unlimited (selected) |
| *Maximum TLS Version* | Unlimited (selected) |
| *Syslog Protocol* | UDP (selected) |
| *Syslog Server* | 192.168.100.39 |
| *Syslog Level* | EXTRA DEBUG (selected) |
| *Send SIP Log* | Yes (checked) |
| *Automatic Reboot* | No (checked) |
| *Automatic Reboot Daily Hour* | 1 |
| *Automatic Reboot Weekly Day* | 1 |
| *Automatic Reboot Monthly Day* | 1 |



### FXS Port Configuration
Based on the provided HTML for the Grandstream HT813's "FXS PORT" configuration page, here is the extracted information:

## Grandstream HT813 FXS Port Configuration

### Account Settings (FXS Port 1)

| Setting | Value | Default/Note |
| :--- | :--- | :--- |
| **Account Active** | Yes (checked) | |
| **Primary SIP Server** | 192.168.100.39 | |
| **Failover SIP Server** | (Empty) | |
| **Prefer Primary SIP Server** | Yes (checked) | |
| **Outbound Proxy** | (Empty) | |
| **Backup Outbound Proxy** | (Empty) | |
| **Prefer Primary Outbound Proxy** | No (checked) | |
| **Allow DHCP Option 120** | No (checked) | |
| **SIP Transport** | UDP (checked) | TCP / TLS |
| **SIP URI Scheme When Using TLS** | sips (checked) | sip / sips |
| **Use Actual Ephemeral Port in Contact with TCP/TLS** | No (checked) | |
| **NAT Traversal** | No (checked) | Keep-Alive / STUN / UPnP |
| **SIP User ID** | 1001 | |
| **Authenticate ID** | 1001 | |
| **Authenticate Password** | (Empty Password Field) | |
| **Name** | 2001pass | |

### SIP Registration and Timers

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *DNS Mode* | A Record (checked) | SRV / NAPTR/SRV |
| *DNS SRV use Registered IP* | No (checked) | |
| *Tel URI* | Disabled (selected) | User=Phone / Enabled |
| *SIP Registration* | Yes (checked) | |
| *Unregister On Reboot* | No (checked) | |
| *Outgoing Call without Registration* | Yes (checked) | |
| *Register Expiration* | 1 minute | (max 45 days) |
| *Reregister before Expiration* | 0 seconds | (0-64800) |
| *SIP Registration Failure Retry Wait Time* | 20 seconds | (1-3600) |
| *SIP Registration Failure Retry Wait Time upon 403 Forbidden* | 120 seconds | (0-3600) |
| *Enable SIP OPTIONS Keep Alive* | No (checked) | |
| *SIP OPTIONS Keep Alive Interval* | 30 seconds | (1-64800) |
| *SIP OPTIONS Keep Alive Max Lost* | 3 | (3-10) |
| *SIP T1 Timeout* | 0.5 sec (selected) | |
| *SIP T2 Interval* | 4 sec (selected) | |
| *SIP Timer D* | 0 | (0-64 seconds) |

### QoS and Ports

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *SIP DSCP (Layer 3 QoS)* | 26 | (0-63, default 26) |
| *RTP DSCP (Layer 3 QoS)* | 46 | (0-63, default 46) |
| *Local SIP Port* | 5060 | (default 5060/5061) |
| *Local RTP Port* | 5004 | (default 5004) |
| *Use Random SIP Port* | No (checked) | |
| *Use Random RTP Port* | No (checked) | |
| *Enable RTCP* | Yes (checked) | |

### Call Features and SIP Handling

| Setting | Value |
| :--- | :--- |
| *Hold Target Before Refer* | Yes (checked) |
| *Refer-To Use Target Contact* | No (checked) |
| *Transfer on Conference Hangup* | No (checked) |
| *Disable Bellcore Style 3-Way Conference* | No (checked) |
| *Remove OBP from Route Header* | No (checked) |
| *Support SIP Instance ID* | Yes (checked) |
| *Validate Incoming SIP Message* | No (checked) |
| *Check SIP User ID for incoming INVITE* | No (checked) |
| *Authenticate incoming INVITE* | No (checked) |
| *Authenticate server certificate domain* | No (checked) |
| *Authenticate server certificate chain* | No (checked) |
| *Allow Incoming SIP Messages from SIP Proxy Only* | No (checked) |
| *Use Privacy Header* | Default (checked) |
| *Use P-Preferred-Identity Header* | Default (checked) |
| *Use P-Access-Network-Info Header* | Yes (checked) |
| *Use P-Emergency-Info Header* | Yes (checked) |
| *SIP REGISTER Contact Header Uses* | LAN Address (checked) |
| *Caller ID Fetch Order* | Auto (checked) |
| *Allow SIP Factory Reset* | No (checked) |
| *DTMF Payload Type* | 101 |
| *Preferred DTMF method (Priority 1)* | RFC2833 (selected) |
| *Preferred DTMF method (Priority 2)* | SIP INFO (selected) |
| *Preferred DTMF method (Priority 3)* | In-audio (selected) |
| *Inband DTMF Duration* | 100 ms |
| *Inband DTMF Inter-duration* | 50 ms |
| *Inband DTMF Tx Gain* | 0 dB | (-12 to 12 dB) |
| *DSP DTMF Detector Duration Threshold* | 30 ms | (20-200 ms) |
| *DSP DTMF Detector Inter-duration* | 30 ms | (20-200 ms) |
| *Disable DTMF Negotiation* | No (checked) |
| *Send Hook Flash Event* | No (checked) |
| *Flash Digit Control* | No (checked) |
| *Enable Call Features* | Yes (checked) |
| *Offhook Auto-Dial* | (Empty) | |
| *Offhook Auto-Dial Delay* | 0 seconds | (0-60) |
| *Proxy-Require* | (Empty) | |
| *Use NAT IP* | (Empty) | |
| *SIP User-Agent* | (Empty) | |
| *SIP User-Agent Postfix* | (Empty) | |
| *Disable Call-Waiting* | No (checked) | |
| *Disable Call-Waiting Caller ID* | No (checked) | |
| *Disable Call-Waiting Tone* | No (checked) | |
| *Disable Connected Line ID* | No (checked) | |
| *Disable Receiver Offhook Tone* | No (checked) | |
| *Disable Reminder Ring for On-Hold Call* | No (checked) | |
| *Disable Visual MWI* | No (checked) | |
| *Do Not Escape '#' as %23 in SIP URI* | No (checked) | |
| *Disable Multiple m line in SDP* | No (checked) | |
| *Ring Timeout* | 60 seconds | (0-300) |
| *Delayed Call Forward Wait Time* | 20 seconds | (1-120) |
| *No Key Entry Timeout* | 4 seconds | (1-15) |
| *Early Dial* | No (checked) | |
| *Dial Plan Prefix* | (Empty) | |
| *Use # as Dial Key* | Yes (checked) | |
| *Dial Plan* | { x+ \| \+x+ \| \*x+ \| \*xx\*x+ } | |
| *SUBSCRIBE for MWI* | No, do not send SUBSCRIBE (checked) | |
| *Send Anonymous* | No (checked) | |
| *Anonymous Call Rejection* | No (checked) | |
| *Special Feature* | Standard (selected) | |
| *Enable Session Timer* | No (checked) | |
| *Session Expiration* | 180 seconds | (90-64800) |
| *Min-SE* | 90 seconds | (90-64800) |
| *Caller Request Timer* | No (checked) | |
| *Callee Request Timer* | No (checked) | |
| *Force Timer* | No (checked) | |
| *UAC Specify Refresher* | Omit (Recommended) (checked) | |
| *UAS Specify Refresher* | UAC (checked) | |
| *Force INVITE* | No (checked) | |
| *When To Restart Session After Re-INVITE received* | Immediately (checked) | |
| *Enable 100rel* | No (checked) | |
| *Add Auth Header On Initial REGISTER* | No (checked) | |
| *Conference URI* | (Empty) | |

### Codec and Media Settings

| Setting | Value |
| :--- | :--- |
| *Use First Matching Vocoder in 200OK SDP* | No (checked) |
| *Preferred Vocoder (Choice 1)* | PCMU (selected) |
| *Preferred Vocoder (Choice 2)* | PCMA (selected) |
| *Preferred Vocoder (Choice 3)* | G723 (selected) |
| *Preferred Vocoder (Choice 4)* | G729 (selected) |
| *Preferred Vocoder (Choice 5)* | G726-32 (selected) |
| *Preferred Vocoder (Choice 6)* | iLBC (selected) |
| *Preferred Vocoder (Choice 7)* | OPUS (selected) |
| *Voice Frames per TX* | 2 |
| *G723 Rate* | 6.3kbps encoding rate (checked) |
| *iLBC Frame Size* | 20ms (checked) |
| *Disable OPUS Stereo in SDP* | No (checked) |
| *iLBC Payload Type* | 97 |
| *OPUS Payload Type* | 123 |
| *VAD* | No (checked) |
| *Symmetric RTP* | No (checked) |
| *Fax Mode* | T.38 (checked) |
| *Re-INVITE After Fax Tone Detected* | Enabled (checked) |
| *Jitter Buffer Type* | Adaptive (checked) |
| *Jitter Buffer Length* | Medium (checked) |
| *SRTP Mode* | Disabled (checked) |
| *SRTP Key Length* | AES 128&256 bit (selected) |
| *Crypto Life Time* | Enabled (checked) |

### FXS Line Settings

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *SLIC Setting* | USA 1 (BELLCORE 600 ohms) (selected) | |
| *Caller ID Scheme* | Bellcore/Telcordia (selected) | |
| *DTMF Caller ID Start Tone* | Default (selected) | |
| *DTMF Caller ID Stop Tone* | Default (selected) | |
| *Polarity Reversal* | Yes (checked) | |
| *Loop Current Disconnect* | No (checked) | |
| *Play busy/reorder tone before Loop Current Disconnect* | No (checked) | |
| *Loop Current Disconnect Duration* | 200 ms | (100 - 10000 ms) |
| *Enable Pulse Dialing* | No (checked) | |
| *Pulse Dialing Standard* | General Standard (selected) | |
| *Enable Hook Flash* | Yes (checked) | |
| *Hook Flash Timing (minimum)* | 300 ms | (40-2000 ms) |
| *Hook Flash Timing (maximum)* | 1100 ms | (40-2000 ms) |
| *On Hook Timing* | 400 ms | (40-2000 ms) |
| *Gain (TX)* | 0dB default (selected) | |
| *Gain (RX)* | -6dB default (selected) | |
| *Disable Line Echo Canceller (LEC)* | No (checked) | |
| *Disable Network Echo Suppressor* | No (checked) | |
| *Outgoing Call Duration Limit* | 0 minutes | (0-180, 0=No Limit) |
| *Incoming Call Duration Limit* | 0 minutes | (0-180, 0=No Limit) |
| *Ring Frequency* | 20Hz default (selected) | |
| *Enable High Ring Power* | No (checked) | |
| *RFC2833 Events Count* | 8 | (2-10) |
| *RFC2833 End Events Count* | 3 | (2-10) |

### Distinctive Ring Tones

| Ring Tone | Used if incoming caller ID is |
| :--- | :--- |
| Ring Tone 1 (selected) | (Empty) |
| Ring Tone 1 (selected) | (Empty) |
| Ring Tone 1 (selected) | (Empty) |

**Ring Tone Cadences (c=on/off)**

| Ring Tone | Cadence |
| :--- | :--- |
| Ring Tone 1 | c=2000/4000; |
| Ring Tone 2 | c=2000/4000; |
| Ring Tone 3 | c=2000/4000; |
| Ring Tone 4 | c=2000/4000; |
| Ring Tone 5 | c=2000/4000; |
| Ring Tone 6 | c=2000/4000; |
| Ring Tone 7 | c=2000/4000; |
| Ring Tone 8 | c=2000/4000; |
| Ring Tone 9 | c=2000/4000; |
| Ring Tone 10 | c=2000/4000; |


### FXO Port Configuration
Based on the provided HTML for the Grandstream HT813's "FXO PORT" configuration page, here is the extracted information:

## Grandstream HT813 FXO Port Configuration

### Account Settings (FXO Port 1)

| Setting | Value | Default/Note |
| :--- | :--- | :--- |
| **Account Active** | Yes (checked) | |
| **Primary SIP Server** | 192.168.100.39 | |
| **Failover SIP Server** | (Empty) | |
| **Prefer Primary SIP Server** | Yes (checked) | |
| **Outbound Proxy** | (Empty) | |
| **Backup Outbound Proxy** | (Empty) | |
| **Prefer Primary Outbound Proxy** | No (checked) | |
| **SIP Transport** | UDP (checked) | TCP / TLS |
| **SIP URI Scheme When Using TLS** | sips (checked) | sip / sips |
| **NAT Traversal** | No (checked) | Keep-Alive / STUN / UPnP |
| **SIP User ID** | 1002 | |
| **Authenticate ID** | 1002 | |
| **Authenticate Password** | (Empty Password Field) | |
| **Name** | HT813 FXO | |

### SIP Registration and Timers

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *DNS Mode* | A Record (checked) | SRV / NAPTR/SRV |
| *DNS SRV use Registered IP* | No (checked) | |
| *Tel URI* | Disabled (selected) | User=Phone / Enabled |
| *SIP Registration* | Yes (checked) | |
| *Unregister On Reboot* | No (checked) | |
| *Outgoing Call without Registration* | Yes (checked) | |
| *Register Expiration* | 1 minute | (max 45 days) |
| *Reregister before Expiration* | 0 seconds | (0-64800) |
| *SIP Registration Failure Retry Wait Time* | 20 seconds | (1-3600) |
| *SIP Registration Failure Retry Wait Time upon 403 Forbidden* | 120 seconds | (0-3600) |
| *Enable SIP OPTIONS Keep Alive* | No (checked) | |
| *SIP OPTIONS Keep Alive Interval* | 30 seconds | (1-64800) |
| *SIP OPTIONS Keep Alive Max Lost* | 3 | (3-10) |
| *SIP T1 Timeout* | 0.5 sec (selected) | |
| *SIP T2 Interval* | 4 sec (selected) | |
| *SIP Timer D* | 0 | (0-64 seconds) |
| *Session Expiration* | 180 seconds | (90-64800) |
| *Min-SE* | 90 seconds | (90-64800) |
| *Caller Request Timer* | No (checked) | |
| *Callee Request Timer* | No (checked) | |
| *Force Timer* | No (checked) | |
| *UAC Specify Refresher* | Omit (Recommended) (checked) | |
| *UAS Specify Refresher* | UAC (checked) | |
| *Force INVITE* | No (checked) | |
| *INVITE Ring-No-Answer Timeout* | 40 seconds | (5-300) |
| *When To Restart Session After Re-INVITE received* | Immediately (checked) | |
| *Enable 100rel* | No (checked) | |
| *Add Auth Header On Initial REGISTER* | No (checked) | |

### QoS and Ports

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *SIP DSCP (Layer 3 QoS)* | 26 | (0-63, default 26) |
| *RTP DSCP (Layer 3 QoS)* | 46 | (0-63, default 46) |
| *Local SIP Port* | 5062 | (default 5062) |
| *Local RTP Port* | 5012 | (default 5012) |
| *Use Random SIP Port* | No (checked) | |
| *Use Random RTP Port* | No (checked) | |
| *Enable RTCP* | Yes (checked) | |

### Call Features and SIP Handling

| Setting | Value |
| :--- | :--- |
| *Remove OBP from Route Header* | No (checked) |
| *Support SIP Instance ID* | Yes (checked) |
| *Validate Incoming SIP Message* | No (checked) |
| *Check SIP User ID for incoming INVITE* | No (checked) |
| *Authenticate incoming INVITE* | No (checked) |
| *Authenticate server certificate domain* | No (checked) |
| *Authenticate server certificate chain* | No (checked) |
| *Allow Incoming SIP Messages from SIP Proxy Only* | No (checked) |
| *Use Privacy Header* | Default (checked) |
| *Use P-Preferred-Identity Header* | Default (checked) |
| *Use P-Access-Network-Info Header* | Yes (checked) |
| *Use P-Emergency-Info Header* | Yes (checked) |
| *SIP REGISTER Contact Header Uses* | LAN Address (checked) |
| *Allow SIP Factory Reset* | No (checked) |
| *DTMF Payload Type* | 101 |
| *Preferred DTMF method (Priority 1)* | RFC2833 (selected) |
| *Preferred DTMF method (Priority 2)* | SIP INFO (selected) |
| *Preferred DTMF method (Priority 3)* | In-audio (selected) |
| *Inband DTMF Duration* | 100 ms |
| *Inband DTMF Inter-duration* | 50 ms |
| *Inband DTMF Tx Gain* | 0 dB |
| *DSP DTMF Detector Duration Threshold* | 30 ms |
| *DSP DTMF Detector Inter-duration* | 30 ms |
| *Disable DTMF Negotiation* | No (checked) |
| *Flash Digit Control* | No (checked) |
| *Proxy-Require* | (Empty) |
| *Use NAT IP* | (Empty) |
| *SIP User-Agent* | (Empty) |
| *SIP User-Agent Postfix* | (Empty) |
| *Do Not Escape '#' as %23 in SIP URI* | No (checked) |
| *Disable Multiple m line in SDP* | No (checked) |
| *Ring Timeout* | 60 seconds | (0-300) |
| *Early Dial* | No (checked) |
| *Dial Plan Prefix* | (Empty) |
| *Use # as Dial Key* | Yes (checked) |
| *Dial Plan* | { x+ \| \+x+ \| \*x+ \| \*xx\*x+ } |
| *SUBSCRIBE for MWI* | No, do not send SUBSCRIBE (checked) |
| *Anonymous Call Rejection* | No (checked) |
| *Special Feature* | Standard (selected) |

### Codec and Media Settings

| Setting | Value |
| :--- | :--- |
| *Use First Matching Vocoder in 200OK SDP* | No (checked) |
| *Preferred Vocoder (Choice 1)* | PCMU (selected) |
| *Preferred Vocoder (Choice 2)* | PCMA (selected) |
| *Preferred Vocoder (Choice 3)* | G723 (selected) |
| *Preferred Vocoder (Choice 4)* | G729 (selected) |
| *Preferred Vocoder (Choice 5)* | G726-32 (selected) |
| *Preferred Vocoder (Choice 6)* | iLBC (selected) |
| *Preferred Vocoder (Choice 7)* | OPUS (selected) |
| *Voice Frames per TX* | 2 |
| *G723 Rate* | 6.3kbps encoding rate (checked) |
| *iLBC Frame Size* | 20ms (checked) |
| *Disable OPUS Stereo in SDP* | No (checked) |
| *iLBC Payload Type* | 97 |
| *OPUS Payload Type* | 123 |
| *VAD* | No (checked) |
| *Symmetric RTP* | No (checked) |
| *Fax Mode* | T.38 (checked) |
| *Re-INVITE After Fax Tone Detected* | Enabled (checked) |
| *Jitter Buffer Type* | Adaptive (checked) |
| *Jitter Buffer Length* | Medium (checked) |
| *SRTP Mode* | Disabled (checked) |
| *SRTP Key Length* | AES 128&256 bit (selected) |
| *Crypto Life Time* | Enabled (checked) |

### FXO Line Settings (PSTN Termination)

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *Caller ID Scheme* | Bellcore/Telcordia (selected) | |
| *DTMF Caller ID Start Tone* | Default (selected) | |
| *DTMF Caller ID Stop Tone* | Default (selected) | |
| *FSK Caller ID Minimum RX Level* | -40 dB | (-96 - 0 dB) |
| *FSK Caller ID Seizure Bits* | 70 | (0 - 800 bits) |
| *FSK Caller ID Mark Bits* | 40 | (1 - 800 bits) |
| *Caller ID Transport Type* | Relay via SIP From (selected) | |
| *Send Hook Flash To PSTN* | Yes (checked) | |
| *Hook Flash Duration* | 600 ms | (200 - 1500 ms) |
| *Gain (TX)* | 0dB default (selected) | |
| *Gain (RX)* | -2dB default (selected) | |
| *Disable Line Echo Canceller (LEC)* | No (checked) | |
| *Disable Network Echo Suppressor* | No (checked) | |
| *Outgoing Call Duration Limit* | 0 minutes | (0-180, 0=No Limit) |
| *Incoming Call Duration Limit* | 0 minutes | (0-180, 0=No Limit) |
| *RFC2833 Events Count* | 8 | (2-10) |
| *RFC2833 End Events Count* | 3 | (2-10) |
| *Enable Current Disconnect* | Yes (checked) | |
| *Current Disconnect Threshold* | 100 ms | (50-800 ms) |
| *Enable PSTN Disconnect Tone Detection* | Yes (checked) | |
| *PSTN Disconnect Tone* | f1=480@-32,f2=620@-32,c=500/500; | |
| *Enable Polarity Reversal* | Yes (checked) | |
| *AC Termination Model* | Country-based (checked) | Impedance-based / Auto-Detected |
| *Country-based* | USA (selected) | |
| *Impedance-based* | 600R -- 600 ohms (selected) | |
| *Number of Rings* | 4 | (1-50) |
| *PSTN Ring Thru FXS* | Yes (checked) | |
| *PSTN Ring Thru Delay* | 4 seconds | (1-10 seconds) |
| *PSTN Ring Timeout* | 6 seconds | (2-10 seconds) |
| *PSTN Idle Wait Timeout between Outgoing Calls* | 4 seconds | (0-10 seconds) |

### Channel Dialing

| Setting | Value | Default/Range |
| :--- | :--- | :--- |
| *DTMF Digit Length* | 100 ms | (40-127 ms) |
| *DTMF Dial Pause* | 100 ms | (40-127 ms) |
| *First Digit Timeout* | 10 seconds | (1-20 seconds) |
| *Inter-Digit Timeout* | 4 seconds | (1-15 seconds) |
| *Wait for Dial-Tone* | Yes (checked) | |
| *Stage Method (1/2)* | 1 | (Default 2 - 2 stage dialing) |
| *Min Delay Before Dial PSTN Number* | 500 ms | (50 - 65000 ms) |