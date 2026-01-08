import asyncio
import os
import logging
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from beautyai_inference.services.email.email_service import EmailService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("🚀 Testing MSAL Email Service...")
    
    # Check for required environment variables
    required_vars = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please export them before running this script:")
        print("export AZURE_TENANT_ID=...")
        print("export AZURE_CLIENT_ID=...")
        print("export AZURE_CLIENT_SECRET=...")
        return

    recipient = os.getenv("TEST_EMAIL_RECIPIENT", "info@gmai.sa")
    print(f"📧 Sending test email to: {recipient}")

    service = EmailService()
    
    try:
        result = await service.send_email(
            to_address=recipient,
            subject="Test Email from GMAI.sa (Graph API)",
            html_body="<h1>It Works!</h1><p>This email was sent using Microsoft Graph API and MSAL.</p>",
            tag="test_script"
        )
        
        if result.get("success"):
            print(f"✅ Email sent successfully! Message ID: {result.get('message_id')}")
        else:
            print(f"❌ Email failed: {result.get('error')}")
            if "details" in result:
                print(f"Details: {result['details']}")
                
    except Exception as e:
        print(f"❌ Exception occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
