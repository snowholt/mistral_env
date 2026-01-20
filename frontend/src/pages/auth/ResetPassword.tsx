/**
 * Reset Password Page
 * 
 * Set new password using token from email.
 */

import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Loader2, Eye, EyeOff, Lock, CheckCircle, XCircle } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useToast } from '@/hooks/use-toast';
import { useAuth, isApiError } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import logo from '@/assets/logo.png';

const resetPasswordSchema = z.object({
  password: z.string().min(8, 'كلمة المرور يجب أن تكون 8 أحرف على الأقل / Password must be at least 8 characters'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: 'كلمات المرور غير متطابقة / Passwords do not match',
  path: ['confirmPassword'],
});

type ResetPasswordFormValues = z.infer<typeof resetPasswordSchema>;

const translations = {
  en: {
    title: 'Reset Password',
    subtitle: 'Enter your new password',
    password: 'New Password',
    passwordPlaceholder: '••••••••',
    confirmPassword: 'Confirm New Password',
    confirmPasswordPlaceholder: '••••••••',
    resetPassword: 'Reset Password',
    resetting: 'Resetting...',
    backToLogin: 'Back to login',
    successTitle: 'Password Reset',
    successDesc: 'Your password has been reset successfully. You can now log in.',
    goToLogin: 'Go to Login',
    error: 'Failed to reset password',
    invalidToken: 'Invalid or Expired Link',
    invalidTokenDesc: 'This password reset link is invalid or has expired. Please request a new one.',
    requestNew: 'Request New Link',
  },
  ar: {
    title: 'إعادة تعيين كلمة المرور',
    subtitle: 'أدخل كلمة المرور الجديدة',
    password: 'كلمة المرور الجديدة',
    passwordPlaceholder: '••••••••',
    confirmPassword: 'تأكيد كلمة المرور الجديدة',
    confirmPasswordPlaceholder: '••••••••',
    resetPassword: 'إعادة تعيين كلمة المرور',
    resetting: 'جاري إعادة التعيين...',
    backToLogin: 'العودة لتسجيل الدخول',
    successTitle: 'تم إعادة تعيين كلمة المرور',
    successDesc: 'تم إعادة تعيين كلمة المرور بنجاح. يمكنك الآن تسجيل الدخول.',
    goToLogin: 'الذهاب لتسجيل الدخول',
    error: 'فشل إعادة تعيين كلمة المرور',
    invalidToken: 'رابط غير صالح أو منتهي الصلاحية',
    invalidTokenDesc: 'هذا الرابط غير صالح أو منتهي الصلاحية. يرجى طلب رابط جديد.',
    requestNew: 'طلب رابط جديد',
  },
};

export default function ResetPassword() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const { resetPassword } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const { language, isRTL } = useLanguage();

  const t = translations[language as keyof typeof translations] || translations.en;

  const form = useForm<ResetPasswordFormValues>({
    resolver: zodResolver(resetPasswordSchema),
    defaultValues: {
      password: '',
      confirmPassword: '',
    },
  });

  // If no token, show error
  if (!token) {
    return (
      <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4 ${isRTL ? 'rtl' : 'ltr'}`}>
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1 text-center">
            <div className="flex justify-center mb-4">
              <div className="h-16 w-16 rounded-full bg-red-100 flex items-center justify-center">
                <XCircle className="h-8 w-8 text-red-600" />
              </div>
            </div>
            <CardTitle className="text-2xl font-bold">{t.invalidToken}</CardTitle>
            <CardDescription>{t.invalidTokenDesc}</CardDescription>
          </CardHeader>
          <CardFooter className="flex justify-center">
            <Link to="/forgot-password">
              <Button>{t.requestNew}</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    );
  }

  const onSubmit = async (data: ResetPasswordFormValues) => {
    setIsLoading(true);
    try {
      await resetPassword(token, data.password);
      setIsSuccess(true);
    } catch (error) {
      const message = isApiError(error) ? error.detail : t.error;
      toast({
        title: t.error,
        description: message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (isSuccess) {
    return (
      <div className={`min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4 ${isRTL ? 'rtl' : 'ltr'}`}>
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1 text-center">
            <div className="flex justify-center mb-4">
              <div className="h-16 w-16 rounded-full bg-green-100 flex items-center justify-center">
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
            </div>
            <CardTitle className="text-2xl font-bold">{t.successTitle}</CardTitle>
            <CardDescription>{t.successDesc}</CardDescription>
          </CardHeader>
          <CardFooter className="flex justify-center">
            <Link to="/login">
              <Button>{t.goToLogin}</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    );
  }

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
                    {t.resetting}
                  </>
                ) : (
                  t.resetPassword
                )}
              </Button>
            </form>
          </Form>
        </CardContent>

        <CardFooter className="flex justify-center">
          <Link to="/login" className="text-sm text-muted-foreground hover:text-primary">
            {t.backToLogin}
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
