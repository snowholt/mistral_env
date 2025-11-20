The information extracted from the HTML, specifically the voice status and statistics, is contained within the JavaScript variable `voiceList`.

Here is the extracted and organized data:

## Voice Line Status and Statistics

| Category | Field | Line 1 | Line 2 |
|---|---|---|---|
| **Voice Interfaces** | | Line 1 | Line 2 |
| **Status** | Admin State | Disabled | Disabled |
| | Phone Number | +966114874423 | 2001 |
| | Registration Status | Disabled | Disabled |
| | Call Status | Idle | Idle |
| | Hook State | On Hook | On Hook |
| **RTP Statistics** | Packets Sent | 0 | 0 |
| | Packets Received | 0 | 0 |
| | Bytes Sent | 0 | 0 |
| | Bytes Received | 0 | 0 |
| | Packets Lost | 0 | 0 |
| **Incoming Calls** | Received | 0 | 0 |
| | Answered | 0 | 0 |
| | Connected | 0 | 0 |
| | Failed | 0 | 0 |
| **Outgoing Calls** | Attempted | 0 | 0 |
| | Answered | 0 | 0 |
| | Connected | 0 | 0 |
| | Failed | 0 | 0 |

**Summary:**

Both **Line 1** and **Line 2** are currently **Disabled** and **Idle**. All packet and call statistics are at **zero (0)**.




---

### SIP
The provided HTML contains configuration settings for the Voice SIP client, primarily stored in JavaScript variables.

Here is the extracted and organized information:

## Voice SIP Configuration Parameters

### 1. Basic SIP Settings

| Field | Value | Description |
|---|---|---|
| **Voice Running Status** | `1` | SIP client is running (based on `voiceRunning` variable). |
| **Bound Interface Name** | `brvlan11` | The network interface used for SIP. |
| **Bound Interface IP** | `20.159.4.1` | The IP address of the bound interface. |
| **Locale Selection** | `KSA` | Saudi Arabia. |
| **Domain Name Mode** | `DomainName` | The SIP domain is specified by name, not IP (based on `am0 = "DomainName"`). |
| **SIP Domain Name** | `fmc.stc.com.sa` | The SIP domain name (`de0`). |
| **Address Mode** | `DomainName` | (`am0`) |
| **Use SIP (Proxy/Registrar)** | Enabled (Implied by `enblPlar.checked = false` and `enblProxy.checked = true` in `frmLoad`) |
| **Use SIP PLAR** | Disabled (Implied by `enblPlar.checked = false` in `frmLoad`) |

#### SIP Proxy/Registrar Information

| Field | Value | Variable |
|---|---|---|
| **SIP Proxy** | `fmc.stc.com.sa` | `pa0` |
| **SIP Proxy Port** | `5060` | `pp0` |
| **SIP Outbound Proxy** | `10.200.42.121` | `oa0` |
| **SIP Outbound Proxy Port** | `5060` | `op0` |
| **SIP Secondary Outbound Proxy** | `0.0.0.0` | `soa0` |
| **SIP Registrar** | `fmc.stc.com.sa` | `ra0` |
| **SIP Registrar Port** | `5060` | `rp0` |

#### SIP PLAR (Private Line Automatic Ringdown) Information

| Field | Value | Variable |
|---|---|---|
| **SIP PLAR Gateway** | `0.0.0.0` | `ga0` |
| **SIP PLAR Port** | `5060` | `gp0` |



### 2. Advanced SIP Settings

| Field | Value | Variable |
|---|---|---|
| **Enable T38 support** | Disabled (`vbd0 = "1"` means `t38Enable.checked = false`) | `vbd0` |
| **WAN MAC + 1** | Enabled (`wm0 = "1"`) | `wm0` |
| **Registration Expire Timeout** | `3600` | `rt0` |
| **Head Start Value (secs)** | `15` | `rrp0` |
| **Registration Retry Interval** | `30` | `rrt0` |
| **DSCP for SIP** | `40` | `tosSip0` |
| **DSCP for RTP** | `46` | `tosRtp0` |
| **Dtmf Relay setting** | `RFC2833` | `dr0` |
| **Hook Flash Relay setting** | `None` | `hr0` |
| **SIP Transport protocol** | `UDP` | `trp0` |
| **MWI Subscribe** | Enabled (`sm0 = "1"`) | `sm0` |
| **Switch Model** | `SoftX3000` | `swt0` |
| **InterDigit Delay** | `5` | `idto0` |
| **Conference URI** | (Empty) | `confuri0` |
| **Conference Option** | `Local` | `confopt0` |

#### Dial Plan Setting

| Field | Value | Variable |
|---|---|---|
| **Voip Dial Plan Setting** | `[2-6]xxxxxx|[7-8]0[1-9]xxxx|[7-8][1-9]xxxxx|05xxxxxxxx|0[123467][2-8]xxxxxx.T|01[123467][2-8]xxxxxx|00xxxxxx.S|1800xx|08111xxxxxx|9[034689]x|700xxxxx.T|800xxxxxxx|92xxxxxxx|1xx.T|**xx|*xx#|*xx*x.#|*xx*x.*x.#|*xx*x.*x.*x.#|*#xx*x.#|*#xx*x.*x.#|*#xx#|#xx#|#xx*x.*x.#` | `dm0` |


### Lines:
The provided HTML contains configuration settings for individual Voice SIP lines (Line 1 and Line 2), primarily stored in JavaScript variables.

Here is the extracted and organized information:

## Voice SIP Line Configurations

The configuration is for **2 lines** (`maxLines = 2`).

### Line 1 Configuration

| Category | Field | Value | Variable |
|---|---|---|---|
| **Account Status** | Admin State (Line Enabled) | Disabled (`le0_0 = "0"`) | `le0_0` |
| | User ID (Extension) | `+966114874423` | `ex0_0` |
| | SIP PLAR User Name | `4003` | `pu0_0` |
| | Display name | `966114874423` | `dn0_0` |
| | Authentication name | `+966114874423@fmc.stc.com.sa` | `an0_0` |
| | Password | `PaSsWoRd` | `pw0_0` |
| **Audio/Codec** | Voice Sample Size (ms) | `20` | `vp0_0` |
| | Silence Suppression (VAD) | Disabled (`ve0_0 = "0"`) | `ve0_0` |
| | Echo Cancellation | Enabled (`ec0_0 = "1"`) | `ec0_0` |
| | Tx Path Gain (Ingress Gain) | `-3 dB` | `ig0_0` |
| | Rx Path Gain (Egress Gain) | `-9 dB` | `eg0_0` |
| **Codec Preference** | Preferred Codecs List | `G.711ALaw,G.711MuLaw,G.729a,G.726_32` | `cl0_0` |
| **Call Features** | Caller ID | Enabled (Implied by `ci0_0 = "0"`) | `ci0_0` |
| | Call Waiting | Enabled (`cw0_0 = "1"`) | `cw0_0` |
| | Centrex | Enabled (`ctx0_0 = "1"`) | `ctx0_0` |
| | Three-way Calling | Enabled (`tw0_0 = "on"`) | `tw0_0` |
| | Message Waiting (MWI) | Enabled (`mw0_0 = "1"`) | `mw0_0` |
| | Reverse Polarity | Disabled (`rpl0_0 = "0"`) | `rpl0_0` |
| | Phone Follows WAN | Disabled (`pfw0_0 = "0"`) | `pfw0_0` |
| **Hot/Warm Line** | Hot/Warm Line Service | `Off` | `wle0_0` |
| | Hot/Warm Line Number | (Empty) | `wln0_0` |
| | Warm Line Timer (ms) | `200` | `wlt0_0` |
| **Dialing** | PBX Dialing | Disabled (`pbx0_0 = "0"`) | `pbx0_0` |
| | Number for PBX Dialing | `9` | `pbn0_0` |
| | E.164 Dialing | Disabled (`e1640_0 = "0"`) | `e1640_0` |
| | International Prefix | `011` | `eidd0_0` |
| | Local Prefix | `1` | `eldp0_0` |
| | Country Code | `1` | `ecc0_0` |

***

### Line 2 Configuration

| Category | Field | Value | Variable |
|---|---|---|---|
| **Account Status** | Admin State (Line Enabled) | Disabled (`le0_1 = "0"`) | `le0_1` |
| | User ID (Extension) | `2001` | `ex0_1` |
| | SIP PLAR User Name | `4004` | `pu0_1` |
| | Display name | `Line2` | `dn0_1` |
| | Authentication name | (Empty) | `an0_1` |
| | Password | `PaSsWoRd` | `pw0_1` |
| **Audio/Codec** | Voice Sample Size (ms) | `20` | `vp0_1` |
| | Silence Suppression (VAD) | Disabled (`ve0_1 = "0"`) | `ve0_1` |
| | Echo Cancellation | Enabled (`ec0_1 = "1"`) | `ec0_1` |
| | Tx Path Gain (Ingress Gain) | `-3 dB` | `ig0_1` |
| | Rx Path Gain (Egress Gain) | `-9 dB` | `eg0_1` |
| **Codec Preference** | Preferred Codecs List | `G.711ALaw,G.711MuLaw,G.729a,G.726_32` | `cl0_1` |
| **Call Features** | Caller ID | Enabled (Implied by `ci0_1 = "0"`) | `ci0_1` |
| | Call Waiting | Enabled (`cw0_1 = "1"`) | `cw0_1` |
| | Centrex | Enabled (`ctx0_1 = "1"`) | `ctx0_1` |
| | Three-way Calling | Enabled (`tw0_1 = "on"`) | `tw0_1` |
| | Message Waiting (MWI) | Enabled (`mw0_1 = "1"`) | `mw0_1` |
| | Reverse Polarity | Disabled (`rpl0_1 = "0"`) | `rpl0_1` |
| | Phone Follows WAN | Disabled (`pfw0_1 = "0"`) | `pfw0_1` |
| **Hot/Warm Line** | Hot/Warm Line Service | `Off` | `wle0_1` |
| | Hot/Warm Line Number | (Empty) | `wln0_1` |
| | Warm Line Timer (ms) | `200` | `wlt0_1` |
| **Dialing** | PBX Dialing | Disabled (`pbx0_1 = "0"`) | `pbx0_1` |
| | Number for PBX Dialing | `9` | `pbn0_1` |
| | E.164 Dialing | Disabled (`e1640_1 = "0"`) | `e1640_1` |
| | International Prefix | `011` | `eidd0_1` |
| | Local Prefix | `1` | `eldp0_1` |
| | Country Code | `1` | `ecc0_1` |



