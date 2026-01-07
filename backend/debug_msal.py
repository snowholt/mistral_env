import asyncio
import os
import logging
import sys
import httpx
import msal
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("🔍 Debugging MSAL Connection...")
    
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    target_user = os.getenv("SMTP_SENDER", "info@gmai.sa")

    if not all([tenant_id, client_id, client_secret]):
        print("❌ Missing credentials")
        return

    app = msal.ConfidentialClientApplication(
        client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        client_credential=client_secret,
    )

    print("1️⃣  Acquiring Token...")
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    
    if "access_token" not in result:
        print(f"❌ Token acquisition failed: {result.get('error_description')}")
        return
    
    token = result['access_token']
    print("✅ Token acquired!")
    
    # Decode token to see roles
    try:
        parts = token.split('.')
        payload = json.loads(base64.b64decode(parts[1] + "==").decode('utf-8'))
        roles = payload.get('roles', [])
        scp = payload.get('scp', "")
        
        print(f"🔑 Token Roles (Application Permissions): {roles}")
        print(f"🔑 Token Scopes (Delegated Permissions): {scp}")
        
        if "Mail.Send" not in roles:
            print("⚠️  WARNING: 'Mail.Send' role is MISSING from Application Permissions!")
            if "Mail.Send" in scp:
                print("💡 DIAGNOSIS: You have 'Mail.Send' as a DELEGATED permission. You MUST remove it and add it as an APPLICATION permission.")
            else:
                print("💡 DIAGNOSIS: Permission is missing or Admin Consent is NOT granted.")
    except Exception as e:
        print(f"⚠️  Could not decode token: {e}")

    print(f"\n2️⃣  Checking User: {target_user}...")
    async with httpx.AsyncClient() as client:
        # Try to get user details
        resp = await client.get(
            f"https://graph.microsoft.com/v1.0/users/{target_user}",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if resp.status_code == 200:
            user_data = resp.json()
            print(f"✅ User found: {user_data.get('displayName')} (ID: {user_data.get('id')})")
            print(f"   UPN: {user_data.get('userPrincipalName')}")
        else:
            print(f"❌ User check failed ({resp.status_code})")
            print(f"   Response: {resp.text}")
            print("\n💡 TIP: The email address in 'SMTP_SENDER' must match the 'User Principal Name' in Azure AD exactly.")

if __name__ == "__main__":
    asyncio.run(main())
