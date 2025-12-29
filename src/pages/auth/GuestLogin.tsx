import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useLanguage } from "@/hooks/useLanguage";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Key, AlertCircle, ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import logo from "@/assets/logo.png";

export default function GuestLogin() {
  const navigate = useNavigate();
  const { language } = useLanguage();
  const { guestLogin } = useAuth();
  const [accessToken, setAccessToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
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

      // Redirect to dashboard
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
            {language === 'ar' ? 'تسجيل دخول الضيف' : 'Guest Access'}
          </h1>
          <p className="text-muted-foreground mt-2">
            {language === 'ar' 
              ? 'أدخل رمز الوصول الذي تلقيته عبر البريد الإلكتروني'
              : 'Enter the access token you received via email'
            }
          </p>
        </div>

        {/* Login Card */}
        <Card>
          <CardHeader>
            <CardTitle>
              {language === 'ar' ? 'الوصول التجريبي' : 'Demo Access'}
            </CardTitle>
            <CardDescription>
              {language === 'ar'
                ? 'استخدم رمز الوصول الخاص بك للبدء'
                : 'Use your access token to get started'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Error Alert */}
              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Access Token Input */}
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
                <p className="text-xs text-muted-foreground">
                  {language === 'ar'
                    ? 'يمكنك العثور على رمز الوصول في البريد الإلكتروني الذي أرسلناه لك'
                    : 'You can find your access token in the email we sent you'
                  }
                </p>
              </div>

              {/* Submit Button */}
              <Button 
                type="submit" 
                className="w-full" 
                disabled={loading}
              >
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
            </form>

            {/* Help Text */}
            <div className="mt-6 p-4 bg-muted/50 rounded-lg space-y-2">
              <p className="text-sm font-medium">
                {language === 'ar' ? 'لم تتلق رمز الوصول؟' : "Haven't received your access token?"}
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
                    ? 'قد يستغرق وصول البريد الإلكتروني بضع دقائق'
                    : 'Email delivery may take a few minutes'
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
