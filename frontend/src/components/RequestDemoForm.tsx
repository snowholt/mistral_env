import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Send, Sparkles, CheckCircle } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { demoApi } from "@/lib/api";

interface DemoRequestFormData {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  company: string;
  companySize: string;
  message: string;
}

const RequestDemoForm = () => {
  const { language } = useLanguage();
  const { toast } = useToast();
  
  const [formData, setFormData] = useState<DemoRequestFormData>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    company: "",
    companySize: "",
    message: ""
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      await demoApi.submitDemoRequest({
        first_name: formData.firstName,
        last_name: formData.lastName,
        email: formData.email,
        phone: formData.phone || undefined,
        company: formData.company || undefined,
        company_size: formData.companySize || undefined,
        message: formData.message || undefined,
      });

      setIsSuccess(true);
      
      toast({
        title: language === 'ar' ? "✅ تم إرسال طلبك بنجاح!" : "✅ Demo request submitted!",
        description: language === 'ar' 
          ? "سنراجع طلبك خلال 24-48 ساعة وسنرسل لك رابط الدخول عبر البريد الإلكتروني."
          : "We'll review your request within 24-48 hours and send you access via email.",
      });

      // Reset form
      setFormData({
        firstName: "",
        lastName: "",
        email: "",
        phone: "",
        company: "",
        companySize: "",
        message: ""
      });

      // Reset success message after 5 seconds
      setTimeout(() => setIsSuccess(false), 5000);
      
    } catch (error: any) {
      console.error('Demo request error:', error);
      
      const errorMessage = error?.detail || error?.message;
      
      // Check for duplicate request error
      if (errorMessage && errorMessage.includes("already have")) {
        toast({
          title: language === 'ar' ? "طلب موجود مسبقاً" : "Request Already Exists",
          description: language === 'ar' 
            ? "لديك طلب تجربة قيد المراجعة أو تمت الموافقة عليه بالفعل. يرجى التحقق من بريدك الإلكتروني."
            : "You already have a pending or approved demo request. Please check your email.",
          variant: "destructive"
        });
      } else {
        toast({
          title: language === 'ar' ? "حدث خطأ" : "Submission Error",
          description: language === 'ar' 
            ? "حدث خطأ عند إرسال الطلب. الرجاء المحاولة مرة أخرى."
            : `There was an error submitting your request. Please try again.`,
          variant: "destructive"
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (field: keyof DemoRequestFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // Show success card
  if (isSuccess) {
    return (
      <section id="request-demo" className="py-20 bg-gradient-to-br from-primary/5 via-transparent to-primary/10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-2xl">
          <Card className="shadow-elegant border-primary/20">
            <CardContent className="pt-16 pb-12 text-center">
              <div className="mb-6">
                <CheckCircle className="h-20 w-20 text-primary mx-auto animate-pulse" />
              </div>
              <h2 className="text-3xl font-bold text-foreground mb-4">
                {language === 'ar' ? "🎉 تم إرسال طلبك بنجاح!" : "🎉 Request Submitted Successfully!"}
              </h2>
              <p className="text-lg text-muted-foreground mb-6">
                {language === 'ar' 
                  ? "شكراً لاهتمامك! سنراجع طلبك ونرسل لك رابط الوصول إلى التجربة المجانية عبر البريد الإلكتروني خلال 24-48 ساعة."
                  : "Thank you for your interest! We'll review your request and send you the free trial access link via email within 24-48 hours."}
              </p>
              <p className="text-sm text-muted-foreground">
                {language === 'ar' 
                  ? "📧 تحقق من بريدك الإلكتروني (بما في ذلك مجلد البريد العشوائي)"
                  : "📧 Check your email (including spam folder)"}
              </p>
            </CardContent>
          </Card>
        </div>
      </section>
    );
  }

  return (
    <section id="request-demo" className="py-20 bg-gradient-to-br from-primary/5 via-transparent to-primary/10">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center mb-4">
            <Sparkles className="h-8 w-8 text-primary animate-pulse" />
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {language === 'ar' ? "جرّب المنصة مجاناً" : "Try Our Platform Free"}
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            {language === 'ar' 
              ? "احصل على وصول مجاني لتجربة محادثة صوتية ذكية بالذكاء الاصطناعي. قدم طلبك الآن!"
              : "Get free access to try our AI-powered voice conversation platform. Submit your request now!"}
          </p>
        </div>

        <Card className="shadow-elegant border-primary/20">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-primary" />
              {language === 'ar' ? "طلب تجربة مجانية" : "Request Free Demo"}
            </CardTitle>
            <CardDescription>
              {language === 'ar' 
                ? "املأ النموذج أدناه وسنرسل لك رابط الوصول خلال 24-48 ساعة"
                : "Fill out the form below and we'll send you access within 24-48 hours"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName">
                    {language === 'ar' ? "الاسم الأول" : "First Name"} *
                  </Label>
                  <Input
                    id="firstName"
                    required
                    value={formData.firstName}
                    onChange={(e) => handleInputChange("firstName", e.target.value)}
                    placeholder={language === 'ar' ? "أدخل اسمك الأول" : "Enter your first name"}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">
                    {language === 'ar' ? "الاسم الأخير" : "Last Name"} *
                  </Label>
                  <Input
                    id="lastName"
                    required
                    value={formData.lastName}
                    onChange={(e) => handleInputChange("lastName", e.target.value)}
                    placeholder={language === 'ar' ? "أدخل اسمك الأخير" : "Enter your last name"}
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="email">
                    {language === 'ar' ? "البريد الإلكتروني" : "Email"} *
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    required
                    value={formData.email}
                    onChange={(e) => handleInputChange("email", e.target.value)}
                    placeholder={language === 'ar' ? "أدخل بريدك الإلكتروني" : "Enter your email address"}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">
                    {language === 'ar' ? "رقم الهاتف" : "Phone Number"}
                  </Label>
                  <Input
                    id="phone"
                    type="tel"
                    value={formData.phone}
                    onChange={(e) => handleInputChange("phone", e.target.value)}
                    placeholder={language === 'ar' ? "أدخل رقم الجوال" : "Enter your phone number"}
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="company">
                    {language === 'ar' ? "اسم الشركة" : "Company Name"}
                  </Label>
                  <Input
                    id="company"
                    value={formData.company}
                    onChange={(e) => handleInputChange("company", e.target.value)}
                    placeholder={language === 'ar' ? "أدخل اسم الشركة" : "Enter your company name"}
                    disabled={isSubmitting}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="companySize">
                    {language === 'ar' ? "حجم الشركة" : "Company Size"}
                  </Label>
                  <Select 
                    value={formData.companySize} 
                    onValueChange={(value) => handleInputChange("companySize", value)}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={language === 'ar' ? "اختر حجم الشركة" : "Select company size"} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1-10">{language === 'ar' ? "1-10 موظف" : "1-10 employees"}</SelectItem>
                      <SelectItem value="11-50">{language === 'ar' ? "11-50 موظف" : "11-50 employees"}</SelectItem>
                      <SelectItem value="51-200">{language === 'ar' ? "51-200 موظف" : "51-200 employees"}</SelectItem>
                      <SelectItem value="201-1000">{language === 'ar' ? "201-1000 موظف" : "201-1000 employees"}</SelectItem>
                      <SelectItem value="1000+">{language === 'ar' ? "1000+ موظف" : "1000+ employees"}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="message">
                  {language === 'ar' ? "رسالتك (اختياري)" : "Your Message (Optional)"}
                </Label>
                <Textarea
                  id="message"
                  value={formData.message}
                  onChange={(e) => handleInputChange("message", e.target.value)}
                  placeholder={language === 'ar' 
                    ? "أخبرنا عن احتياجاتك أو أي أسئلة لديك..."
                    : "Tell us about your needs or any questions you have..."}
                  rows={4}
                  disabled={isSubmitting}
                />
              </div>

              <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                <p className="text-sm text-muted-foreground">
                  <strong className="text-foreground">
                    {language === 'ar' ? "ماذا ستحصل:" : "What you'll get:"}
                  </strong>
                  <br />
                  {language === 'ar' 
                    ? "• وصول مجاني للتجربة لمدة 7 أيام • ما يصل إلى 10 محادثات صوتية • وصول كامل للذكاء الاصطناعي الصوتي"
                    : "• Free trial access for 7 days • Up to 10 voice conversations • Full AI voice-to-voice access"}
                </p>
              </div>

              <Button 
                type="submit" 
                variant="cta" 
                size="lg" 
                className="w-full group" 
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <>
                    <Send className="mr-2 h-5 w-5 animate-pulse" />
                    {language === 'ar' ? "جار الإرسال..." : "Submitting..."}
                  </>
                ) : (
                  <>
                    <Send className="mr-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                    {language === 'ar' ? "إرسال الطلب" : "Submit Request"}
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default RequestDemoForm;
