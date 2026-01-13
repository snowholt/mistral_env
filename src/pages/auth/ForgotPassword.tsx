/**
 * Forgot Password Page
 * 
 * Request password reset email.
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Loader2, Mail, ArrowLeft, CheckCircle } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useToast } from '@/hooks/use-toast';
import { useAuth, isApiError } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import logo from '@/assets/logo.png';

const forgotPasswordSchema = z.object({
  email: z.string().email('البريد الإلكتروني غير صالح / Invalid email'),
});

type ForgotPasswordFormValues = z.infer<typeof forgotPasswordSchema>;

const translations = {
  en: {
    title: 'Forgot Password',
    subtitle: 'Enter your email and we\'ll send you a reset link',
    email: 'Email',
    emailPlaceholder: 'you@example.com',
    sendLink: 'Send Reset Link',
    sending: 'Sending...',
    backToLogin: 'Back to login',
    successTitle: 'Check your email',
    successDesc: 'We\'ve sent a password reset link to your email address.',
    error: 'Failed to send reset link',
  },
  ar: {
    title: 'نسيت كلمة المرور',
    subtitle: 'أدخل بريدك الإلكتروني وسنرسل لك رابط إعادة التعيين',
    email: 'البريد الإلكتروني',
    emailPlaceholder: 'you@example.com',
    sendLink: 'إرسال رابط إعادة التعيين',
    sending: 'جاري الإرسال...',
    backToLogin: 'العودة لتسجيل الدخول',
    successTitle: 'تحقق من بريدك الإلكتروني',
    successDesc: 'لقد أرسلنا رابط إعادة تعيين كلمة المرور إلى بريدك الإلكتروني.',
    error: 'فشل إرسال رابط إعادة التعيين',
  },
};

export default function ForgotPassword() {
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const { forgotPassword } = useAuth();
  const { toast } = useToast();
  const { language, isRTL } = useLanguage();

  const t = translations[language as keyof typeof translations] || translations.en;

  const form = useForm<ForgotPasswordFormValues>({
    resolver: zodResolver(forgotPasswordSchema),
    defaultValues: {
      email: '',
    },
  });

  const onSubmit = async (data: ForgotPasswordFormValues) => {
    setIsLoading(true);
    try {
      await forgotPassword(data.email);
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
              <Button variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                {t.backToLogin}
              </Button>
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

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t.sending}
                  </>
                ) : (
                  t.sendLink
                )}
              </Button>
            </form>
          </Form>
        </CardContent>

        <CardFooter className="flex justify-center">
          <Link to="/login" className="text-sm text-muted-foreground hover:text-primary flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" />
            {t.backToLogin}
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
