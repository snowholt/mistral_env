/**
 * Login Page
 * 
 * User authentication with email and password.
 * Bilingual support (Arabic/English).
 */

import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Loader2, Eye, EyeOff, Mail, Lock } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useToast } from '@/hooks/use-toast';
import { useAuth, isApiError } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import logo from '@/assets/logo.png';

const loginSchema = z.object({
  email: z.string().email('البريد الإلكتروني غير صالح / Invalid email'),
  password: z.string().min(6, 'كلمة المرور يجب أن تكون 6 أحرف على الأقل / Password must be at least 6 characters'),
});

type LoginFormValues = z.infer<typeof loginSchema>;

const translations = {
  en: {
    title: 'Welcome Back',
    subtitle: 'Sign in to your account to continue',
    email: 'Email',
    emailPlaceholder: 'you@example.com',
    password: 'Password',
    passwordPlaceholder: '••••••••',
    forgotPassword: 'Forgot password?',
    signIn: 'Sign In',
    signingIn: 'Signing in...',
    noAccount: "Don't have an account?",
    signUp: 'Sign up',
    backToHome: 'Back to home',
    loginSuccess: 'Welcome back!',
    loginError: 'Login failed',
  },
  ar: {
    title: 'مرحباً بعودتك',
    subtitle: 'سجل الدخول إلى حسابك للمتابعة',
    email: 'البريد الإلكتروني',
    emailPlaceholder: 'you@example.com',
    password: 'كلمة المرور',
    passwordPlaceholder: '••••••••',
    forgotPassword: 'نسيت كلمة المرور؟',
    signIn: 'تسجيل الدخول',
    signingIn: 'جاري تسجيل الدخول...',
    noAccount: 'ليس لديك حساب؟',
    signUp: 'إنشاء حساب',
    backToHome: 'العودة للرئيسية',
    loginSuccess: 'مرحباً بعودتك!',
    loginError: 'فشل تسجيل الدخول',
  },
};

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const location = useLocation();
  const { language, isRTL } = useLanguage();

  const t = translations[language as keyof typeof translations] || translations.en;

  const form = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginFormValues) => {
    setIsLoading(true);
    try {
      await login(data.email, data.password);
      toast({
        title: t.loginSuccess,
      });
      // Redirect to intended page or dashboard
      const from = (location.state as { from?: Location })?.from?.pathname || '/app';
      navigate(from, { replace: true });
    } catch (error) {
      const message = isApiError(error) ? error.detail : t.loginError;
      toast({
        title: t.loginError,
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4 ${isRTL ? 'rtl' : 'ltr'}`}>
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 text-center">
          <Link to="/" className="flex justify-center mb-4">
            <img src={logo} alt="Genius AI" className="h-12" />
          </Link>
          <CardTitle className="text-2xl font-bold">{t.title}</CardTitle>
          <CardDescription>{t.subtitle}</CardDescription>
        </CardHeader>
        
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{t.email}</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Mail className={`absolute top-3 h-4 w-4 text-muted-foreground ${isRTL ? 'right-3' : 'left-3'}`} />
                        <Input
                          type="email"
                          placeholder={t.emailPlaceholder}
                          className={isRTL ? 'pr-10' : 'pl-10'}
                          {...field}
                        />
                      </div>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex items-center justify-between">
                      <FormLabel>{t.password}</FormLabel>
                      <Link
                        to="/forgot-password"
                        className="text-sm text-primary hover:underline"
                      >
                        {t.forgotPassword}
                      </Link>
                    </div>
                    <FormControl>
                      <div className="relative">
                        <Lock className={`absolute top-3 h-4 w-4 text-muted-foreground ${isRTL ? 'right-3' : 'left-3'}`} />
                        <Input
                          type={showPassword ? 'text' : 'password'}
                          placeholder={t.passwordPlaceholder}
                          className={isRTL ? 'pr-10 pl-10' : 'pl-10 pr-10'}
                          {...field}
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className={`absolute top-3 text-muted-foreground hover:text-foreground ${isRTL ? 'left-3' : 'right-3'}`}
                        >
                          {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </button>
                      </div>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t.signingIn}
                  </>
                ) : (
                  t.signIn
                )}
              </Button>
            </form>
          </Form>
        </CardContent>

        <CardFooter className="flex flex-col gap-4">
          <div className="text-sm text-center text-muted-foreground">
            {t.noAccount}{' '}
            <Link to="/register" className="text-primary hover:underline font-medium">
              {t.signUp}
            </Link>
          </div>
          <Link to="/" className="text-sm text-muted-foreground hover:text-primary">
            ← {t.backToHome}
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
