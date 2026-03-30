/**
 * WhatsApp Connect Page
 * 
 * Allows customers to connect their WhatsApp Business accounts
 * via Meta Embedded Signup flow with OTP 2FA verification.
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  MessageSquare,
  Phone,
  CheckCircle2,
  AlertCircle,
  Plus,
  Loader2,
  ExternalLink,
  Trash2,
  Settings,
  RefreshCw,
  Shield,
} from 'lucide-react';
import { api } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import { loadMetaSDK, startWhatsAppSignup } from '@/lib/meta-sdk';
import OTPVerificationModal from '@/components/whatsapp/OTPVerificationModal';
import { useToast } from '@/components/ui/use-toast';

const translations = {
  en: {
    title: 'WhatsApp Connect',
    description: 'Connect your WhatsApp Business account to start receiving customer messages.',
    connectedAccounts: 'Connected Accounts',
    noAccounts: 'No WhatsApp accounts connected',
    noAccountsDescription: 'Connect your first WhatsApp Business account to get started.',
    connectNew: 'Connect WhatsApp Account',
    connecting: 'Connecting...',
    phoneNumber: 'Phone Number',
    status: 'Status',
    active: 'Active',
    inactive: 'Inactive',
    pending: 'Pending',
    disconnect: 'Disconnect',
    configure: 'Configure AI',
    refresh: 'Refresh Status',
    verifyFirst: 'Verify Identity',
    verifyDescription: 'For security, we need to verify your identity via email before connecting WhatsApp.',
    prerequisites: 'Before you connect',
    prereq1: 'A verified Facebook Business account',
    prereq2: 'Access to a phone number for WhatsApp Business',
    prereq3: 'Business verification documents (if not already verified)',
    loadingSDK: 'Loading Meta SDK...',
    sdkError: 'Failed to load Meta SDK. Please refresh and try again.',
    connectSuccess: 'WhatsApp account connected successfully!',
    connectError: 'Failed to connect WhatsApp account.',
    disconnectConfirm: 'Are you sure you want to disconnect this WhatsApp account?',
    disconnectSuccess: 'WhatsApp account disconnected.',
    lastSync: 'Last synced',
    messages: 'Messages',
    today: 'today',
    emailNotVerified: 'Email Not Verified',
    emailNotVerifiedDesc: 'Please verify your email address before connecting WhatsApp. Check your inbox for the verification link.',
    resendVerification: 'Resend Verification Email',
    verificationSent: 'Verification email sent! Check your inbox.',
    verificationSentError: 'Failed to send verification email. Please try again.',
    alreadyVerified: "I've already verified",
    checkingVerification: 'Checking...',
    nowVerified: 'Email verified! You can now connect WhatsApp.',
    stillNotVerified: 'Email not yet verified. Please click the link in your email.',
  },
  ar: {
    title: 'ربط واتساب',
    description: 'اربط حساب واتساب للأعمال لبدء استقبال رسائل العملاء.',
    connectedAccounts: 'الحسابات المتصلة',
    noAccounts: 'لا توجد حسابات واتساب متصلة',
    noAccountsDescription: 'اربط أول حساب واتساب للأعمال للبدء.',
    connectNew: 'ربط حساب واتساب',
    connecting: 'جاري الربط...',
    phoneNumber: 'رقم الهاتف',
    status: 'الحالة',
    active: 'نشط',
    inactive: 'غير نشط',
    pending: 'قيد الانتظار',
    disconnect: 'قطع الاتصال',
    configure: 'تهيئة الذكاء الاصطناعي',
    refresh: 'تحديث الحالة',
    verifyFirst: 'تحقق من الهوية',
    verifyDescription: 'لأمانك، نحتاج للتحقق من هويتك عبر البريد الإلكتروني قبل ربط واتساب.',
    prerequisites: 'قبل الربط',
    prereq1: 'حساب أعمال فيسبوك مُوثّق',
    prereq2: 'رقم هاتف لواتساب للأعمال',
    prereq3: 'مستندات توثيق الأعمال (إن لم تكن مُوثّقة)',
    loadingSDK: 'جاري تحميل Meta SDK...',
    sdkError: 'فشل تحميل Meta SDK. يرجى التحديث والمحاولة مرة أخرى.',
    connectSuccess: 'تم ربط حساب واتساب بنجاح!',
    connectError: 'فشل في ربط حساب واتساب.',
    disconnectConfirm: 'هل أنت متأكد من قطع الاتصال بحساب واتساب هذا؟',
    disconnectSuccess: 'تم قطع اتصال حساب واتساب.',
    lastSync: 'آخر مزامنة',
    messages: 'الرسائل',
    today: 'اليوم',
    emailNotVerified: 'البريد الإلكتروني غير مُوثّق',
    emailNotVerifiedDesc: 'يرجى تأكيد بريدك الإلكتروني قبل ربط واتساب. تحقق من بريدك الوارد لرابط التحقق.',
    resendVerification: 'إعادة إرسال رسالة التحقق',
    verificationSent: 'تم إرسال رسالة التحقق! تحقق من بريدك الوارد.',
    verificationSentError: 'فشل في إرسال رسالة التحقق. حاول مرة أخرى.',
    alreadyVerified: 'لقد أكدت بريدي',
    checkingVerification: 'جاري التحقق...',
    nowVerified: 'تم تأكيد بريدك! يمكنك الآن ربط واتساب.',
    stillNotVerified: 'لم يتم تأكيد البريد بعد. يرجى النقر على الرابط في بريدك.',
  },
};

interface WhatsAppAccount {
  id: number;
  phone_number: string;
  display_phone_number: string;
  verified_name: string;
  quality_rating: string;
  status: 'active' | 'inactive' | 'pending';
  waba_id: string;
  created_at: string;
  last_synced_at: string;
  messages_today: number;
}

export default function WhatsAppConnect() {
  const { user, refreshUser } = useAuth();
  const { language, isRTL } = useLanguage();
  const { toast } = useToast();
  const t = translations[language as keyof typeof translations] || translations.en;

  const [accounts, setAccounts] = useState<WhatsAppAccount[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const [sdkLoaded, setSdkLoaded] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);
  const [showOTPModal, setShowOTPModal] = useState(false);
  const [otpVerified, setOtpVerified] = useState(false);
  const [showVerifyAlert, setShowVerifyAlert] = useState(false);
  const [isResendingVerification, setIsResendingVerification] = useState(false);
  const [isCheckingVerification, setIsCheckingVerification] = useState(false);

  // Load accounts on mount
  useEffect(() => {
    fetchAccounts();
    loadSDK();
  }, []);

  const loadSDK = async () => {
    try {
      await loadMetaSDK();
      setSdkLoaded(true);
    } catch (error) {
      console.error('Failed to load Meta SDK:', error);
      setSdkError(t.sdkError);
    }
  };

  const fetchAccounts = async () => {
    setIsLoading(true);
    try {
      const response = await api.get<{ accounts: WhatsAppAccount[] }>(
        '/api/v1/whatsapp/accounts'
      );
      setAccounts(response.accounts || []);
    } catch (error) {
      console.error('Failed to fetch WhatsApp accounts:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnectClick = () => {
    // Check if email is verified first
    if (!user?.is_verified) {
      setShowVerifyAlert(true);
      return;
    }
    // Show OTP verification modal
    setShowOTPModal(true);
  };

  const handleResendVerification = async () => {
    if (!user?.email) return;
    setIsResendingVerification(true);
    try {
      await api.post('/api/v1/auth/resend-verification', { email: user.email });
      toast({
        title: 'Success',
        description: t.verificationSent,
      });
      setShowVerifyAlert(false);
    } catch (error) {
      console.error('Failed to resend verification:', error);
      toast({
        title: 'Error',
        description: t.verificationSentError,
        variant: 'destructive',
      });
    } finally {
      setIsResendingVerification(false);
    }
  };

  const handleCheckVerification = async () => {
    setIsCheckingVerification(true);
    try {
      const refreshedUser = await refreshUser();
      // Check the freshly returned user data directly
      if (refreshedUser?.is_verified) {
        toast({
          title: 'Success',
          description: t.nowVerified,
        });
        setShowVerifyAlert(false);
        // Auto-open OTP modal since they're now verified
        setShowOTPModal(true);
      } else {
        toast({
          title: 'Info',
          description: t.stillNotVerified,
          variant: 'destructive',
        });
      }
    } catch (error) {
      console.error('Failed to check verification:', error);
    } finally {
      setIsCheckingVerification(false);
    }
  };

  const handleOTPSuccess = () => {
    setOtpVerified(true);
    setShowOTPModal(false);
    // Now proceed with Meta Embedded Signup
    initiateWhatsAppSignup();
  };

  const initiateWhatsAppSignup = async () => {
    if (!sdkLoaded) {
      toast({
        title: 'Error',
        description: t.sdkError,
        variant: 'destructive',
      });
      return;
    }

    setIsConnecting(true);

    try {
      // Get the signup configuration from backend
      const config = await api.get<{ config_id: string; session_id: string }>(
        '/api/v1/whatsapp/signup/init'
      );

      // Start Meta Embedded Signup
      startWhatsAppSignup(
        config.config_id,
        async (code, waba_id, phone_number_id) => {
          // Complete signup on backend
          try {
            await api.post('/api/v1/whatsapp/signup/complete', {
              code,
              session_id: config.session_id,
            });
            
            toast({
              title: 'Success',
              description: t.connectSuccess,
            });
            
            // Refresh accounts list
            fetchAccounts();
          } catch (error) {
            console.error('Failed to complete signup:', error);
            toast({
              title: 'Error',
              description: t.connectError,
              variant: 'destructive',
            });
          } finally {
            setIsConnecting(false);
            setOtpVerified(false);
          }
        },
        (error) => {
          console.error('WhatsApp signup error:', error);
          toast({
            title: 'Error',
            description: error,
            variant: 'destructive',
          });
          setIsConnecting(false);
          setOtpVerified(false);
        }
      );
    } catch (error) {
      console.error('Failed to initiate signup:', error);
      toast({
        title: 'Error',
        description: t.connectError,
        variant: 'destructive',
      });
      setIsConnecting(false);
      setOtpVerified(false);
    }
  };

  const handleDisconnect = async (accountId: number) => {
    if (!confirm(t.disconnectConfirm)) return;

    try {
      await api.delete(`/api/v1/whatsapp/accounts/${accountId}`);
      toast({
        title: 'Success',
        description: t.disconnectSuccess,
      });
      fetchAccounts();
    } catch (error) {
      console.error('Failed to disconnect account:', error);
      toast({
        title: 'Error',
        description: 'Failed to disconnect account',
        variant: 'destructive',
      });
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge className="bg-green-500">{t.active}</Badge>;
      case 'inactive':
        return <Badge variant="secondary">{t.inactive}</Badge>;
      case 'pending':
        return <Badge variant="outline">{t.pending}</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className={`space-y-6 ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t.title}</h1>
        <p className="text-muted-foreground">{t.description}</p>
      </div>

      {/* SDK Error Alert */}
      {sdkError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{sdkError}</AlertDescription>
        </Alert>
      )}

      {/* Email Not Verified Alert */}
      {showVerifyAlert && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>{t.emailNotVerified}</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <span>{t.emailNotVerifiedDesc}</span>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                className="w-fit"
                onClick={handleResendVerification}
                disabled={isResendingVerification || isCheckingVerification}
              >
                {isResendingVerification ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    {t.resendVerification}
                  </>
                )}
              </Button>
              <Button
                variant="default"
                size="sm"
                className="w-fit"
                onClick={handleCheckVerification}
                disabled={isResendingVerification || isCheckingVerification}
              >
                {isCheckingVerification ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t.checkingVerification}
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    {t.alreadyVerified}
                  </>
                )}
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Prerequisites Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            {t.prerequisites}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
              {t.prereq1}
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
              {t.prereq2}
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
              {t.prereq3}
            </li>
          </ul>
        </CardContent>
      </Card>

      {/* Connected Accounts */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>{t.connectedAccounts}</CardTitle>
            <CardDescription>
              {accounts.length} account{accounts.length !== 1 ? 's' : ''} connected
            </CardDescription>
          </div>
          <Button
            onClick={handleConnectClick}
            disabled={!sdkLoaded || isConnecting}
          >
            {isConnecting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t.connecting}
              </>
            ) : !sdkLoaded ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t.loadingSDK}
              </>
            ) : (
              <>
                <Plus className="mr-2 h-4 w-4" />
                {t.connectNew}
              </>
            )}
          </Button>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : accounts.length === 0 ? (
            <div className="text-center py-8">
              <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="font-medium">{t.noAccounts}</h3>
              <p className="text-sm text-muted-foreground mt-1">
                {t.noAccountsDescription}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {accounts.map((account) => (
                <div
                  key={account.id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div className="flex items-center gap-4">
                    <div className="h-12 w-12 bg-green-100 rounded-full flex items-center justify-center">
                      <Phone className="h-6 w-6 text-green-600" />
                    </div>
                    <div>
                      <div className="font-medium">
                        {account.verified_name || account.display_phone_number}
                      </div>
                      <div className="text-sm text-muted-foreground flex items-center gap-2">
                        <span>{account.display_phone_number}</span>
                        <span>•</span>
                        {getStatusBadge(account.status)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {t.messages}: {account.messages_today} {t.today}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <a href={`/app/whatsapp/settings?account=${account.id}`}>
                        <Settings className="h-4 w-4 mr-1" />
                        {t.configure}
                      </a>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDisconnect(account.id)}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* OTP Verification Modal */}
      <OTPVerificationModal
        isOpen={showOTPModal}
        onClose={() => setShowOTPModal(false)}
        onSuccess={handleOTPSuccess}
        userEmail={user?.email || ''}
        purpose="whatsapp_connect"
      />
    </div>
  );
}
