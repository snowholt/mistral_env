import { useState, useEffect } from "react";
import { useNavigate, Link, useSearchParams } from "react-router-dom";
import { useLanguage } from "@/hooks/useLanguage";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Key, AlertCircle, ArrowLeft, Mail, Lock, CheckCircle, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";
import logo from "@/assets/logo.png";

// API base URL
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface TokenValidationResult {
  valid: boolean;
  email?: string;
  expires_at?: string;
  days_remaining?: number;
  max_conversations?: number;
  error?: string;
}

interface PasswordRequirements {
  min_length: number;
  max_length: number;
  requirements: string[];
}

type ViewMode = 'loading' | 'setup-password' | 'login' | 'legacy-token';

export default function GuestLogin() {
  const navigate = useNavigate();
  const { language } = useLanguage();
  const { guestLogin, guestPasswordLogin } = useAuth();
  const [searchParams] = useSearchParams();
  
  // View mode state
  const [viewMode, setViewMode] = useState<ViewMode>('loading');
  
  // Token validation state
  const [tokenEmail, setTokenEmail] = useState<string>('');
  const [daysRemaining, setDaysRemaining] = useState<number>(0);
  const [setupToken, setSetupToken] = useState<string>('');
  
  // Form states
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [accessToken, setAccessToken] = useState("");
  
  // Loading and error states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [passwordRequirements, setPasswordRequirements] = useState<PasswordRequirements | null>(null);
  
  // Check for token in URL on mount
  useEffect(() => {
    const token = searchParams.get('token');
    if (token) {
      setSetupToken(token);
      validateSetupToken(token);
    } else {
      setViewMode('login');
    }
    
    // Fetch password requirements
    fetchPasswordRequirements();
  }, [searchParams]);
  
  const fetchPasswordRequirements = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/auth/guest/password-requirements`);
      if (response.ok) {
        const data = await response.json();
        setPasswordRequirements(data);
      }
    } catch (err) {
      console.error("Failed to fetch password requirements:", err);
    }
  };
  
  const validateSetupToken = async (token: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/auth/guest/validate-setup-token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      
      const data: TokenValidationResult = await response.json();
      
      if (data.valid && data.email) {
        setTokenEmail(data.email);
        setDaysRemaining(data.days_remaining || 0);
        setViewMode('setup-password');
      } else {
        setError(data.error || (language === 'ar' 
          ? 'رابط غير صالح أو منتهي الصلاحية' 
          : 'Invalid or expired activation link'));
        setViewMode('login');
      }
    } catch (err) {
      console.error("Token validation error:", err);
      setError(language === 'ar'
        ? 'حدث خطأ في التحقق من الرابط'
        : 'Error validating activation link');
      setViewMode('login');
    } finally {
      setLoading(false);
    }
  };
  
  const handleSetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    
    if (password !== confirmPassword) {
      setError(language === 'ar' ? 'كلمات المرور غير متطابقة' : 'Passwords do not match');
      return;
    }
    
    if (password.length < 8) {
      setError(language === 'ar' 
        ? 'كلمة المرور يجب أن تكون 8 أحرف على الأقل' 
        : 'Password must be at least 8 characters');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/auth/guest/set-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: setupToken,
          password,
          confirm_password: confirmPassword,
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        const errorMsg = typeof data.detail === 'object' 
          ? data.detail.message 
          : data.detail;
        throw new Error(errorMsg || 'Failed to set password');
      }

      // Immediately log in so the auth context is updated (prevents redirects to /login).
      // This also persists the correct tokens in localStorage via the shared guest auth flow.
      const loginEmail: string = (data?.guest_user?.email || tokenEmail || '').trim();
      if (loginEmail) {
        await guestPasswordLogin(loginEmail, password);
      }
      
      toast.success(
        language === 'ar'
          ? 'تم تفعيل حسابك بنجاح! مرحباً بك'
          : 'Account activated successfully! Welcome'
      );
      
      navigate('/app');
    } catch (err: any) {
      console.error("Set password error:", err);
      setError(err.message || (language === 'ar'
        ? 'فشل في تعيين كلمة المرور'
        : 'Failed to set password'));
    } finally {
      setLoading(false);
    }
  };
  
  const handlePasswordLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    
    if (!email.trim() || !password.trim()) {
      setError(language === 'ar'
        ? 'الرجاء إدخال البريد الإلكتروني وكلمة المرور'
        : 'Please enter email and password');
      return;
    }
    
    setLoading(true);
    
    try {
      await guestPasswordLogin(email.trim(), password);
      
      toast.success(
        language === 'ar'
          ? 'مرحباً! تم تسجيل الدخول بنجاح'
          : 'Welcome! Successfully logged in'
      );
      
      navigate('/app');
    } catch (err: any) {
      console.error("Guest login error:", err);
      const errorMessage = err?.response?.data?.detail || 
        (language === 'ar'
          ? 'بريد إلكتروني أو كلمة مرور غير صحيحة'
          : 'Invalid email or password');
      
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const handleLegacyTokenLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!accessToken.trim()) {
      setError(
        language === 'ar'
          ? 'الرجاء إدخال رمز الوصول'
          : 'Please enter your access token'
      );
      return;
    }

    setLoading(true);

    try {
      await guestLogin(accessToken.trim());
      
      toast.success(
        language === 'ar'
          ? 'مرحباً! تم تسجيل الدخول بنجاح'
          : 'Welcome! Successfully logged in'
      );

      navigate('/app');
    } catch (err: any) {
      console.error("Guest login error:", err);
      const errorMessage = err?.response?.data?.detail || 
        (language === 'ar'
          ? 'رمز وصول غير صالح أو منتهي الصلاحية'
          : 'Invalid or expired access token');
      
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Loading state while checking token
  if (viewMode === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex items-center justify-center p-4">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">
            {language === 'ar' ? 'جاري التحقق...' : 'Verifying...'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Back to Home */}
        <Link 
          to="/" 
          className="inline-flex items-center text-sm text-muted-foreground hover:text-primary mb-6 transition-colors"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          {language === 'ar' ? 'العودة للرئيسية' : 'Back to Home'}
        </Link>

        {/* Logo */}
        <div className="text-center mb-8">
          <img src={logo} alt="Genius AI" className="h-16 mx-auto mb-4" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            {viewMode === 'setup-password' 
              ? (language === 'ar' ? 'تفعيل الحساب' : 'Activate Account')
              : (language === 'ar' ? 'تسجيل دخول الضيف' : 'Guest Access')
            }
          </h1>
          <p className="text-muted-foreground mt-2">
            {viewMode === 'setup-password'
              ? (language === 'ar' 
                  ? 'قم بتعيين كلمة مرور لتفعيل حسابك'
                  : 'Set a password to activate your account')
              : (language === 'ar' 
                  ? 'ادخل ببريدك الإلكتروني وكلمة المرور'
                  : 'Login with your email and password')
            }
          </p>
        </div>

        {/* Main Card */}
        <Card>
          <CardHeader>
            <CardTitle>
              {viewMode === 'setup-password'
                ? (language === 'ar' ? 'إنشاء كلمة المرور' : 'Create Password')
                : (language === 'ar' ? 'الوصول التجريبي' : 'Demo Access')
              }
            </CardTitle>
            <CardDescription>
              {viewMode === 'setup-password'
                ? (language === 'ar'
                    ? `مرحباً ${tokenEmail}! قم بتعيين كلمة مرور للوصول (${daysRemaining} أيام متبقية)`
                    : `Welcome ${tokenEmail}! Set a password for access (${daysRemaining} days remaining)`)
                : (language === 'ar'
                    ? 'سجل الدخول لبدء العرض التجريبي'
                    : 'Login to start your demo')
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Error Alert */}
            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Setup Password Form */}
            {viewMode === 'setup-password' && (
              <form onSubmit={handleSetPassword} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="password">
                    {language === 'ar' ? 'كلمة المرور' : 'Password'}
                  </Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder={language === 'ar' ? 'أدخل كلمة المرور' : 'Enter password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 pr-10"
                      disabled={loading}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">
                    {language === 'ar' ? 'تأكيد كلمة المرور' : 'Confirm Password'}
                  </Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input
                      id="confirmPassword"
                      type={showPassword ? "text" : "password"}
                      placeholder={language === 'ar' ? 'أعد إدخال كلمة المرور' : 'Re-enter password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="pl-10"
                      disabled={loading}
                    />
                  </div>
                </div>
                
                {/* Password Requirements */}
                {passwordRequirements && (
                  <div className="p-3 bg-muted/50 rounded-lg text-xs space-y-1">
                    <p className="font-medium text-muted-foreground">
                      {language === 'ar' ? 'متطلبات كلمة المرور:' : 'Password requirements:'}
                    </p>
                    <ul className="space-y-1 text-muted-foreground">
                      {passwordRequirements.requirements.map((req, idx) => (
                        <li key={idx} className="flex items-center gap-1">
                          <CheckCircle className="h-3 w-3" />
                          {req}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      {language === 'ar' ? 'جاري التفعيل...' : 'Activating...'}
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-4 w-4 mr-2" />
                      {language === 'ar' ? 'تفعيل الحساب' : 'Activate Account'}
                    </>
                  )}
                </Button>
              </form>
            )}

            {/* Email/Password Login Form */}
            {viewMode === 'login' && (
              <form onSubmit={handlePasswordLogin} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">
                    {language === 'ar' ? 'البريد الإلكتروني' : 'Email'}
                  </Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder={language === 'ar' ? 'أدخل بريدك الإلكتروني' : 'Enter your email'}
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10"
                      disabled={loading}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="loginPassword">
                    {language === 'ar' ? 'كلمة المرور' : 'Password'}
                  </Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input
                      id="loginPassword"
                      type={showPassword ? "text" : "password"}
                      placeholder={language === 'ar' ? 'أدخل كلمة المرور' : 'Enter password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 pr-10"
                      disabled={loading}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                    </button>
                  </div>
                </div>

                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      {language === 'ar' ? 'جاري تسجيل الدخول...' : 'Logging in...'}
                    </>
                  ) : (
                    <>
                      <Key className="h-4 w-4 mr-2" />
                      {language === 'ar' ? 'دخول' : 'Login'}
                    </>
                  )}
                </Button>
                
                {/* Switch to legacy token login */}
                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => setViewMode('legacy-token')}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {language === 'ar' 
                      ? 'لديك رمز وصول قديم؟ انقر هنا' 
                      : 'Have a legacy access token? Click here'}
                  </button>
                </div>
              </form>
            )}

            {/* Legacy Token Login Form */}
            {viewMode === 'legacy-token' && (
              <form onSubmit={handleLegacyTokenLogin} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="accessToken">
                    {language === 'ar' ? 'رمز الوصول' : 'Access Token'}
                  </Label>
                  <div className="relative">
                    <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input
                      id="accessToken"
                      type="text"
                      placeholder={
                        language === 'ar'
                          ? 'الصق رمز الوصول هنا...'
                          : 'Paste your access token here...'
                      }
                      value={accessToken}
                      onChange={(e) => setAccessToken(e.target.value)}
                      className="pl-10"
                      disabled={loading}
                      autoComplete="off"
                    />
                  </div>
                </div>

                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      {language === 'ar' ? 'جاري التحقق...' : 'Verifying...'}
                    </>
                  ) : (
                    <>
                      <Key className="h-4 w-4 mr-2" />
                      {language === 'ar' ? 'دخول' : 'Access Demo'}
                    </>
                  )}
                </Button>
                
                {/* Switch to password login */}
                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => setViewMode('login')}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {language === 'ar' 
                      ? 'العودة لتسجيل الدخول بكلمة المرور' 
                      : 'Back to password login'}
                  </button>
                </div>
              </form>
            )}

            {/* Help Text */}
            {viewMode !== 'setup-password' && (
              <div className="mt-6 p-4 bg-muted/50 rounded-lg space-y-2">
                <p className="text-sm font-medium">
                  {language === 'ar' ? 'لم تتلق رسالة التفعيل؟' : "Haven't received your activation email?"}
                </p>
                <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
                  <li>
                    {language === 'ar'
                      ? 'تحقق من صندوق البريد الوارد والرسائل غير المرغوب فيها'
                      : 'Check your inbox and spam folder'
                    }
                  </li>
                  <li>
                    {language === 'ar'
                      ? 'رابط التفعيل صالح لمدة ساعة واحدة فقط'
                      : 'Activation link is valid for 1 hour only'
                    }
                  </li>
                  <li>
                    {language === 'ar'
                      ? 'تأكد من أن طلب التجربة الخاص بك قد تمت الموافقة عليه'
                      : 'Make sure your demo request has been approved'
                    }
                  </li>
                </ul>
              </div>
            )}

            {/* Contact Support */}
            <div className="mt-4 text-center text-sm text-muted-foreground">
              {language === 'ar' ? 'تحتاج مساعدة؟' : 'Need help?'}{' '}
              <Link 
                to="/#contact" 
                className="text-primary hover:underline font-medium"
              >
                {language === 'ar' ? 'اتصل بالدعم' : 'Contact Support'}
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Regular User Login */}
        <div className="mt-6 text-center text-sm">
          <span className="text-muted-foreground">
            {language === 'ar' ? 'مستخدم مسجل؟' : 'Regular user?'}
          </span>{' '}
          <Link to="/login" className="text-primary hover:underline font-medium">
            {language === 'ar' ? 'تسجيل الدخول هنا' : 'Login here'}
          </Link>
        </div>
      </div>
    </div>
  );
}
