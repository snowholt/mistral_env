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
    
    @classmethod
    def demo_request_confirmation_email(cls, full_name: str) -> str:
        """Generate demo request confirmation email for requester."""
        content = f"""
<div class="content-ar">
    <p class="greeting">شكراً لاهتمامك {full_name}! 🌟</p>
    <p>تم استلام طلبك لتجربة منصة GMAI.sa بنجاح.</p>
    
    <p><strong>الخطوات التالية:</strong></p>
    <ul style="margin: 15px 0; padding-right: 20px;">
        <li>سيقوم فريقنا بمراجعة طلبك خلال 24-48 ساعة</li>
        <li>سنرسل لك بريد إلكتروني يحتوي على رابط الدخول عند الموافقة</li>
        <li>ستتمكن من تجربة محادثة صوتية مع الذكاء الاصطناعي مجاناً</li>
    </ul>
    
    <div class="note">
        📧 راقب بريدك الإلكتروني للحصول على رابط الوصول إلى التجربة المجانية.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Thank you for your interest {full_name}! 🌟</p>
    <p>We've successfully received your demo request for GMAI.sa platform.</p>
    
    <p><strong>Next Steps:</strong></p>
    <ul style="margin: 15px 0; padding-left: 20px;">
        <li>Our team will review your request within 24-48 hours</li>
        <li>You'll receive an email with access link upon approval</li>
        <li>You'll be able to try a free voice conversation with AI</li>
    </ul>
    
    <p><small>Keep an eye on your email for your free demo access link.</small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="تأكيد طلب التجربة - Demo Request Confirmed",
            content=content,
        )
    
    @classmethod
    def demo_request_admin_notification_email(
        cls, 
        requester_name: str,
        requester_email: str,
        company: str,
        company_size: str,
        message: str,
        demo_request_id: int,
        admin_panel_url: str
    ) -> str:
        """Generate admin notification email for new demo request."""
        content = f"""
<div class="content-ar">
    <p class="greeting">طلب تجربة جديد 🔔</p>
    <p>تم استلام طلب تجربة جديد من عميل محتمل.</p>
    
    <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <p style="margin: 5px 0;"><strong>الاسم:</strong> {requester_name}</p>
        <p style="margin: 5px 0;"><strong>البريد الإلكتروني:</strong> {requester_email}</p>
        <p style="margin: 5px 0;"><strong>الشركة:</strong> {company or 'غير محدد'}</p>
        <p style="margin: 5px 0;"><strong>حجم الشركة:</strong> {company_size or 'غير محدد'}</p>
        {f'<p style="margin: 5px 0;"><strong>الرسالة:</strong><br>{message}</p>' if message else ''}
    </div>
    
    <div class="button-container">
        <a href="{admin_panel_url}/admin/demo-requests/{demo_request_id}" class="button">مراجعة الطلب</a>
    </div>
    
    <div class="note">
        💡 يمكنك الموافقة على الطلب أو رفضه من لوحة التحكم.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>New Demo Request 🔔</p>
    <p>A new demo request has been received from a potential customer.</p>
    
    <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <p style="margin: 5px 0;"><strong>Name:</strong> {requester_name}</p>
        <p style="margin: 5px 0;"><strong>Email:</strong> {requester_email}</p>
        <p style="margin: 5px 0;"><strong>Company:</strong> {company or 'Not specified'}</p>
        <p style="margin: 5px 0;"><strong>Company Size:</strong> {company_size or 'Not specified'}</p>
        {f'<p style="margin: 5px 0;"><strong>Message:</strong><br>{message}</p>' if message else ''}
    </div>
    
    <div class="button-container">
        <a href="{admin_panel_url}/admin/demo-requests/{demo_request_id}" class="button">Review Request</a>
    </div>
    
    <p><small>You can approve or reject this request from the admin panel.</small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="طلب تجربة جديد - New Demo Request",
            content=content,
        )
    
    @classmethod
    def demo_access_granted_email(
        cls,
        full_name: str,
        access_token: str,
        login_url: str,
        expires_days: int,
        max_conversations: int,
        activation_hours: int = 72,
    ) -> str:
        """Generate demo access granted email with setup link for account activation."""
        # The access_token is actually the setup_token for the new password flow
        # Build the activation URL with the token
        activation_url = f"{login_url}?token={access_token}"
        
        content = f"""
<div class="content-ar">
    <p class="greeting">مبروك {full_name}! 🎉</p>
    <p>تمت الموافقة على طلبك! يمكنك الآن تجربة منصة GMAI.sa مجاناً.</p>
    
    <div style="background-color: #ecfdf5; border: 2px solid #10B981; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
        <p style="margin: 0 0 15px 0; font-size: 18px;"><strong>🔐 أنشئ كلمة المرور الخاصة بك</strong></p>
        <p style="margin: 10px 0; color: #666;">اضغط على الزر أدناه لإنشاء كلمة مرور وتفعيل حسابك</p>
        <a href="{activation_url}" style="display: inline-block; background-color: #10B981; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold; margin: 15px 0;">تفعيل الحساب 🚀</a>
        <p style="margin: 15px 0 0 0; font-size: 12px; color: #888;">⏰ هذا الرابط صالح لمدة {activation_hours} ساعة فقط</p>
    </div>
    
    <p><strong>حدود التجربة المجانية:</strong></p>
    <ul style="margin: 15px 0; padding-right: 20px;">
        <li>⏰ صالح لمدة {expires_days} أيام</li>
        <li>💬 حتى {max_conversations} محادثة صوتية</li>
        <li>🎤 وصول كامل لمحادثة الصوت إلى الصوت بالذكاء الاصطناعي</li>
    </ul>
    
    <div class="note">
        💡 بعد إنشاء كلمة المرور، يمكنك تسجيل الدخول بالبريد الإلكتروني وكلمة المرور.<br>
        📧 بعد انتهاء التجربة، يمكنك الترقية إلى حساب كامل.
    </div>
</div>

<div class="divider"></div>

<div class="content-en">
    <p>Congratulations {full_name}! 🎉</p>
    <p>Your request has been approved! You can now try GMAI.sa platform for free.</p>
    
    <div style="background-color: #ecfdf5; border: 2px solid #10B981; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
        <p style="margin: 0 0 15px 0; font-size: 18px;"><strong>🔐 Create Your Password</strong></p>
        <p style="margin: 10px 0; color: #666;">Click the button below to create a password and activate your account</p>
        <a href="{activation_url}" style="display: inline-block; background-color: #10B981; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-size: 16px; font-weight: bold; margin: 15px 0;">Activate Account 🚀</a>
        <p style="margin: 15px 0 0 0; font-size: 12px; color: #888;">⏰ This link is valid for {activation_hours} hours only</p>
    </div>
    
    <p><strong>Free Trial Limits:</strong></p>
    <ul style="margin: 15px 0; padding-left: 20px;">
        <li>⏰ Valid for {expires_days} days</li>
        <li>💬 Up to {max_conversations} voice conversations</li>
        <li>🎤 Full access to AI voice-to-voice chat</li>
    </ul>
    
    <p><small>
        💡 After creating your password, you can log in with your email and password.<br>
        📧 After the trial ends, you can upgrade to a full account.
    </small></p>
</div>
"""
        return cls.BASE_TEMPLATE.format(
            title="تفعيل حسابك - Activate Your Account",
            content=content,
        )

