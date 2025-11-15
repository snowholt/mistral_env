#!/bin/bash
# Setup NAT for OpenVPN to access local network
# This script should be run after OpenVPN starts

echo "Setting up NAT for OpenVPN traffic..."

# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# Add MASQUERADE rule for VPN clients to access local network
iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -o enp12s0 -j MASQUERADE

# Save the rules
mkdir -p /etc/iptables
iptables-save > /etc/iptables/rules.v4

echo "✅ NAT rules configured for VPN access to 192.168.100.0/24"
