/**
 * Register Page
 * 
 * User registration with email, password, and optional name.
 * Bilingual support (Arabic/English).
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Loader2, Eye, EyeOff, Mail, Lock, User } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useToast } from '@/hooks/use-toast';
import { useAuth, isApiError } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import logo from '@/assets/logo.png';

const registerSchema = z.object({
  fullName: z.string().min(2, 'الاسم يجب أن يكون حرفين على الأقل / Name must be at least 2 characters'),
  email: z.string().email('البريد الإلكتروني غير صالح / Invalid email'),
  password: z.string().min(8, 'كلمة المرور يجب أن تكون 8 أحرف على الأقل / Password must be at least 8 characters'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: 'كلمات المرور غير متطابقة / Passwords do not match',
  path: ['confirmPassword'],
});

type RegisterFormValues = z.infer<typeof registerSchema>;

const translations = {
  en: {
    title: 'Create Account',
    subtitle: 'Start your free trial today',
    fullName: 'Full Name',
    fullNamePlaceholder: 'Ahmed Mohammed',
    email: 'Email',
    emailPlaceholder: 'you@example.com',
    password: 'Password',
    passwordPlaceholder: '••••••••',
    confirmPassword: 'Confirm Password',
    confirmPasswordPlaceholder: '••••••••',
    createAccount: 'Create Account',
    creating: 'Creating account...',
    hasAccount: 'Already have an account?',
    signIn: 'Sign in',
    backToHome: 'Back to home',
    registerSuccess: 'Account created!',
    registerSuccessDesc: 'Please check your email to verify your account.',
    registerError: 'Registration failed',
  },
  ar: {
    title: 'إنشاء حساب',
    subtitle: 'ابدأ تجربتك المجانية اليوم',
    fullName: 'الاسم الكامل',
    fullNamePlaceholder: 'أحمد محمد',
    email: 'البريد الإلكتروني',
    emailPlaceholder: 'you@example.com',
    password: 'كلمة المرور',
    passwordPlaceholder: '••••••••',
    confirmPassword: 'تأكيد كلمة المرور',
    confirmPasswordPlaceholder: '••••••••',
    createAccount: 'إنشاء حساب',
    creating: 'جاري إنشاء الحساب...',
    hasAccount: 'لديك حساب بالفعل؟',
    signIn: 'تسجيل الدخول',
    backToHome: 'العودة للرئيسية',
    registerSuccess: 'تم إنشاء الحساب!',
    registerSuccessDesc: 'يرجى التحقق من بريدك الإلكتروني لتفعيل حسابك.',
    registerError: 'فشل إنشاء الحساب',
  },
};

export default function Register() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const { language, isRTL } = useLanguage();

  const t = translations[language as keyof typeof translations] || translations.en;

  const form = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      fullName: '',
      email: '',
      password: '',
      confirmPassword: '',
    },
  });

  const onSubmit = async (data: RegisterFormValues) => {
    setIsLoading(true);
    try {
      await register(data.email, data.password, data.fullName);
      toast({
        title: t.registerSuccess,
        description: t.registerSuccessDesc,
      });
      navigate('/login', { state: { registered: true, email: data.email } });
    } catch (error) {
      const message = isApiError(error) ? error.detail : t.registerError;
      toast({
        title: t.registerError,
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
                name="fullName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{t.fullName}</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <User className={`absolute top-3 h-4 w-4 text-muted-foreground ${isRTL ? 'right-3' : 'left-3'}`} />
                        <Input
                          placeholder={t.fullNamePlaceholder}
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
                    <FormLabel>{t.password}</FormLabel>
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

              <FormField
                control={form.control}
                name="confirmPassword"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{t.confirmPassword}</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Lock className={`absolute top-3 h-4 w-4 text-muted-foreground ${isRTL ? 'right-3' : 'left-3'}`} />
                        <Input
                          type={showConfirmPassword ? 'text' : 'password'}
                          placeholder={t.confirmPasswordPlaceholder}
                          className={isRTL ? 'pr-10 pl-10' : 'pl-10 pr-10'}
                          {...field}
                        />
                        <button
                          type="button"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          className={`absolute top-3 text-muted-foreground hover:text-foreground ${isRTL ? 'left-3' : 'right-3'}`}
                        >
                          {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
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
                    {t.creating}
                  </>
                ) : (
                  t.createAccount
                )}
              </Button>
            </form>
          </Form>
        </CardContent>

        <CardFooter className="flex flex-col gap-4">
          <div className="text-sm text-center text-muted-foreground">
            {t.hasAccount}{' '}
            <Link to="/login" className="text-primary hover:underline font-medium">
              {t.signIn}
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
