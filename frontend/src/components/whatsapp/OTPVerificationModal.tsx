/**
 * OTP Verification Modal
 * 
 * 2FA verification modal for sensitive actions like WhatsApp account connection.
 * Sends OTP to user's email and verifies before proceeding.
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import {
  InputOTP,
  InputOTPGroup,
  InputOTPSlot,
} from '@/components/ui/input-otp';
import { Loader2, Mail, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '@/lib/api';
import { useLanguage } from '@/hooks/useLanguage';

const translations = {
  en: {
    title: 'Email Verification',
    description: 'For your security, we\'ve sent a 6-digit code to your email.',
    emailSent: 'Code sent to',
    enterCode: 'Enter the code below:',
    verify: 'Verify',
    verifying: 'Verifying...',
    resend: 'Resend Code',
    resendIn: 'Resend in',
    seconds: 's',
    success: 'Verified successfully!',
    error: 'Invalid or expired code',
    sendError: 'Failed to send code. Please try again.',
    tooManyAttempts: 'Too many attempts. Please wait.',
  },
  ar: {
    title: 'التحقق من البريد',
    description: 'لأمانك، أرسلنا رمزًا مكونًا من 6 أرقام إلى بريدك الإلكتروني.',
    emailSent: 'تم إرسال الرمز إلى',
    enterCode: 'أدخل الرمز أدناه:',
    verify: 'تأكيد',
    verifying: 'جاري التحقق...',
    resend: 'إعادة إرسال الرمز',
    resendIn: 'إعادة الإرسال خلال',
    seconds: 'ث',
    success: 'تم التحقق بنجاح!',
    error: 'رمز غير صالح أو منتهي الصلاحية',
    sendError: 'فشل في إرسال الرمز. حاول مرة أخرى.',
    tooManyAttempts: 'محاولات كثيرة جدًا. يرجى الانتظار.',
  },
};

interface OTPVerificationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  userEmail: string;
  purpose?: 'whatsapp_connect' | 'sensitive_action';
}

export default function OTPVerificationModal({
  isOpen,
  onClose,
  onSuccess,
  userEmail,
  purpose = 'whatsapp_connect',
}: OTPVerificationModalProps) {
  const { language } = useLanguage();
  const t = translations[language as keyof typeof translations] || translations.en;

  const [otpValue, setOtpValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [otpSent, setOtpSent] = useState(false);

  // Request OTP when modal opens
  useEffect(() => {
    if (isOpen && !otpSent) {
      requestOTP();
    }
  }, [isOpen]);

  // Cooldown timer
  useEffect(() => {
    if (resendCooldown > 0) {
      const timer = setTimeout(() => setResendCooldown(resendCooldown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [resendCooldown]);

  // Auto-verify when 6 digits entered
  useEffect(() => {
    if (otpValue.length === 6 && !isLoading && !success) {
      verifyOTP();
    }
  }, [otpValue]);

  const requestOTP = async () => {
    setIsSending(true);
    setError(null);
    
    try {
      await api.post('/api/v1/auth/otp/request', { purpose });
      setOtpSent(true);
      setResendCooldown(60); // 60 second cooldown
    } catch (err: any) {
      if (err.status === 429) {
        setError(t.tooManyAttempts);
      } else {
        setError(t.sendError);
      }
    } finally {
      setIsSending(false);
    }
  };

  const verifyOTP = async () => {
    setIsLoading(true);
    setError(null);

    try {
      await api.post('/api/v1/auth/otp/verify', { code: otpValue, purpose });
      setSuccess(true);
      // Brief delay to show success state
      setTimeout(() => {
        onSuccess();
        handleClose();
      }, 1000);
    } catch (err: any) {
      setError(t.error);
      setOtpValue('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setOtpValue('');
    setError(null);
    setSuccess(false);
    setOtpSent(false);
    onClose();
  };

  const handleResend = () => {
    if (resendCooldown === 0) {
      setOtpValue('');
      requestOTP();
    }
  };

  // Mask email for display
  const maskedEmail = userEmail.replace(
    /^(.{2})(.*)(@.*)$/,
    (_, start, middle, end) => start + '*'.repeat(Math.min(middle.length, 5)) + end
  );

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Mail className="h-5 w-5 text-primary" />
            {t.title}
          </DialogTitle>
          <DialogDescription>
            {t.description}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col items-center space-y-6 py-4">
          {/* Email indicator */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/50 px-4 py-2 rounded-lg">
            <Mail className="h-4 w-4" />
            <span>{t.emailSent} <strong>{maskedEmail}</strong></span>
          </div>

          {/* OTP Input */}
          {!success && (
            <>
              <div className="text-sm font-medium">{t.enterCode}</div>
              <InputOTP
                maxLength={6}
                value={otpValue}
                onChange={setOtpValue}
                disabled={isLoading || isSending}
              >
                <InputOTPGroup>
                  <InputOTPSlot index={0} />
                  <InputOTPSlot index={1} />
                  <InputOTPSlot index={2} />
                  <InputOTPSlot index={3} />
                  <InputOTPSlot index={4} />
                  <InputOTPSlot index={5} />
                </InputOTPGroup>
              </InputOTP>
            </>
          )}

          {/* Success state */}
          {success && (
            <div className="flex items-center gap-2 text-green-600">
              <CheckCircle2 className="h-6 w-6" />
              <span className="font-medium">{t.success}</span>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <XCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          )}

          {/* Actions */}
          {!success && (
            <div className="flex flex-col items-center gap-3 w-full">
              <Button
                onClick={verifyOTP}
                disabled={otpValue.length !== 6 || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t.verifying}
                  </>
                ) : (
                  t.verify
                )}
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={handleResend}
                disabled={resendCooldown > 0 || isSending}
              >
                {isSending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : resendCooldown > 0 ? (
                  `${t.resendIn} ${resendCooldown}${t.seconds}`
                ) : (
                  t.resend
                )}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
