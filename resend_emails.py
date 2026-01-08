#!/usr/bin/env python3
"""Quick script to resend demo access emails."""

import requests
import json

API_BASE = "https://api.gmai.sa"

# Login as admin
print("🔐 Logging in as admin...")
login_resp = requests.post(
    f"{API_BASE}/api/v1/whatsapp/auth/login",
    json={"email": "nariman@gmai.sa", "password": "Admin@123456"}
)
if login_resp.status_code != 200:
    print(f"❌ Login failed: {login_resp.text}")
    exit(1)

token = login_resp.json().get("access_token")
print(f"✅ Got token: {token[:30]}...")

headers = {"Authorization": f"Bearer {token}"}

# Resend email for request ID 2 (snow.holt@gmail.com)
print("\n📧 Resending email to snow.holt@gmail.com (request ID 2)...")
resp1 = requests.post(
    f"{API_BASE}/api/v1/admin/demo-requests/2/resend-access-email",
    headers=headers
)
print(f"   Status: {resp1.status_code}")
print(f"   Response: {resp1.json()}")

# Resend email for request ID 1 (jafari.nariman@gmail.com)
print("\n📧 Resending email to jafari.nariman@gmail.com (request ID 1)...")
resp2 = requests.post(
    f"{API_BASE}/api/v1/admin/demo-requests/1/resend-access-email",
    headers=headers
)
print(f"   Status: {resp2.status_code}")
print(f"   Response: {resp2.json()}")

print("\n✨ Done! Check your emails!")
