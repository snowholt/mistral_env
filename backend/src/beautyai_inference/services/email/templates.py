"""
Email templates for BeautyAI platform.

All templates are bilingual (Arabic primary, English secondary).
PDPL compliant with Saudi Arabia data handling notice.
"""


class EmailTemplates:
    """
    HTML email templates for transactional emails.
    
    Design principles:
    - Bilingual: Arabic (RTL) primary, English secondary
    - Mobile-responsive
    - Clean, professional styling
    - GMAI.sa branding
    """
    
    # Base template with common styles and structure
    BASE_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }}
        .content {{
            padding: 30px 20px;
        }}
        .content-ar {{
            direction: rtl;
            text-align: right;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .content-en {{
            direction: ltr;
            text-align: left;
            color: #666;
            font-size: 14px;
        }}
        .greeting {{
            font-size: 18px;
            margin-bottom: 20px;
        }}
        .button {{
            display: inline-block;
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white !important;
            text-decoration: none;
            padding: 14px 32px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            margin: 20px 0;
        }}
        .button:hover {{
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
        }}
        .button-container {{
            text-align: center;
            margin: 25px 0;
        }}
        .footer {{
            background-color: #f9fafb;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #eee;
        }}
        .footer a {{
            color: #10B981;
            text-decoration: none;
        }}
        .note {{
            background-color: #fef3cd;
            border-right: 4px solid #ffc107;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 14px;
        }}
        .divider {{
            border-top: 1px solid #eee;
            margin: 20px 0;
        }}
        @media only screen and (max-width: 600px) {{
            .container {{
                margin: 10px;
                border-radius: 0;
            }}
            .content {{
                padding: 20px 15px;
            }}
            .header {{
                padding: 20px 15px;
            }}
            .header h1 {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <div style="padding: 20px 0;">
        <div class="container">
            <div class="header">
                <h1>GMAI.sa</h1>
            </div>
            <div class="content">
                {content}
            </div>
            <div class="footer">
                <p>
                    GMAI.sa - منصة الذكاء الاصطناعي للمحادثات<br>
                    AI-Powered Conversation Platform
                </p>
                <p>
                    <a href="https://gmai.sa">gmai.sa</a> | 
                    <a href="https://gmai.sa/privacy-policy">سياسة الخصوصية</a> | 
                    <a href="https://gmai.sa/terms">الشروط والأحكام</a>
                </p>
                <p style="margin-top: 15px; font-size: 11px; color: #999;">
                    هذا البريد مرسل من المملكة العربية السعودية ومتوافق مع نظام حماية البيانات الشخصية (PDPL)<br>
                    This email is sent from Saudi Arabia and is PDPL compliant.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    @classmethod
    def verification_email(cls, full_name: str, verification_url: str) -> str:
        """Generate email verification template."""
        content = f"""
<div class="content-ar">
    <p class="greeting">مرحباً {full_name} 👋</p>
    <p>شكراً لتسجيلك في GMAI.sa! يرجى تأكيد بريدك الإلكتروني للبدء في استخدام المنصة.</p>
    
    <div class="button-container">
        <a href="{verification_url}" class="button">تأكيد البريد الإلكتروني ✓</a>
    </div>
    
    <div class="note">
        ⚠️ هذا الرابط صالح لمدة 24 ساعة فقط. إذا لم تقم بإنشاء هذا الحساب، يمكنك تجاهل هذا البريد.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Hello {full_name},</p>
    <p>Thank you for registering with GMAI.sa! Please verify your email to get started.</p>
    
    <div class="button-container">
        <a href="{verification_url}" class="button">Verify Email ✓</a>
    </div>
    
    <p><small>This link expires in 24 hours. If you didn't create this account, please ignore this email.</small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="تأكيد البريد الإلكتروني - Email Verification",
            content=content,
        )
    
    @classmethod
    def password_reset_email(cls, full_name: str, reset_url: str) -> str:
        """Generate password reset template."""
        content = f"""
<div class="content-ar">
    <p class="greeting">مرحباً {full_name}</p>
    <p>تلقينا طلباً لإعادة تعيين كلمة المرور الخاصة بحسابك. إذا كنت أنت من طلب ذلك، انقر على الزر أدناه:</p>
    
    <div class="button-container">
        <a href="{reset_url}" class="button">إعادة تعيين كلمة المرور 🔐</a>
    </div>
    
    <div class="note">
        ⚠️ هذا الرابط صالح لمدة ساعة واحدة فقط. إذا لم تطلب إعادة تعيين كلمة المرور، يرجى تجاهل هذا البريد - حسابك آمن.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Hello {full_name},</p>
    <p>We received a request to reset your password. If you made this request, click the button below:</p>
    
    <div class="button-container">
        <a href="{reset_url}" class="button">Reset Password 🔐</a>
    </div>
    
    <p><small>This link expires in 1 hour. If you didn't request a password reset, please ignore this email - your account is safe.</small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="إعادة تعيين كلمة المرور - Password Reset",
            content=content,
        )
    
    @classmethod
    def welcome_email(cls, full_name: str, dashboard_url: str) -> str:
        """Generate welcome email after verification."""
        content = f"""
<div class="content-ar">
    <p class="greeting">مرحباً بك {full_name}! 🎉</p>
    <p>تم تأكيد بريدك الإلكتروني بنجاح. أنت الآن جاهز للبدء في استخدام GMAI.sa!</p>
    
    <p><strong>ماذا يمكنك فعله الآن:</strong></p>
    <ul style="margin: 15px 0; padding-right: 20px;">
        <li>ربط حساب واتساب للأعمال الخاص بك</li>
        <li>إعداد مساعد الذكاء الاصطناعي لخدمة عملائك</li>
        <li>إضافة أسئلة شائعة ومعلومات عن شركتك</li>
        <li>اختبار المحادثات قبل الإطلاق</li>
    </ul>
    
    <div class="button-container">
        <a href="{dashboard_url}" class="button">الذهاب إلى لوحة التحكم 🚀</a>
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Welcome {full_name}! 🎉</p>
    <p>Your email has been verified. You're now ready to start using GMAI.sa!</p>
    
    <p><strong>What you can do now:</strong></p>
    <ul style="margin: 15px 0; padding-left: 20px;">
        <li>Connect your WhatsApp Business account</li>
        <li>Set up your AI assistant for customer service</li>
        <li>Add FAQs and company information</li>
        <li>Test conversations before going live</li>
    </ul>
    
    <div class="button-container">
        <a href="{dashboard_url}" class="button">Go to Dashboard 🚀</a>
    </div>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="مرحباً بك - Welcome",
            content=content,
        )
    
    @classmethod
    def admin_invite_email(cls, invite_url: str, invited_by: str) -> str:
        """Generate admin invite email (for @gmai.sa users only)."""
        content = f"""
<div class="content-ar">
    <p class="greeting">دعوة للانضمام كمسؤول 🔑</p>
    <p>لقد تمت دعوتك من قبل <strong>{invited_by}</strong> للانضمام إلى فريق إدارة GMAI.sa.</p>
    
    <p>كمسؤول، ستتمكن من:</p>
    <ul style="margin: 15px 0; padding-right: 20px;">
        <li>عرض وإدارة جميع العملاء</li>
        <li>مراقبة الاستخدام والإيرادات</li>
        <li>إدارة خطط الاشتراك</li>
        <li>الوصول إلى التقارير والتحليلات</li>
    </ul>
    
    <div class="button-container">
        <a href="{invite_url}" class="button">قبول الدعوة ✓</a>
    </div>
    
    <div class="note">
        ⚠️ هذه الدعوة خاصة بعناوين البريد الإلكتروني @gmai.sa فقط.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Admin Invitation 🔑</p>
    <p>You've been invited by <strong>{invited_by}</strong> to join the GMAI.sa admin team.</p>
    
    <p>As an admin, you'll be able to:</p>
    <ul style="margin: 15px 0; padding-left: 20px;">
        <li>View and manage all customers</li>
        <li>Monitor usage and revenue</li>
        <li>Manage subscription plans</li>
        <li>Access reports and analytics</li>
    </ul>
    
    <div class="button-container">
        <a href="{invite_url}" class="button">Accept Invitation ✓</a>
    </div>
    
    <p><small>This invite is exclusive to @gmai.sa email addresses.</small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="دعوة للانضمام كمسؤول - Admin Invitation",
            content=content,
        )
