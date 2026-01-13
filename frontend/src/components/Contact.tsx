import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Mail, Phone, MapPin, Send, MessageCircle } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const Contact = () => {
  const { language } = useLanguage();
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phonenumber: "",
    company: "",
    companySize: "",
    message: ""
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // IMPORTANT: Replace this with your actual Alibaba Cloud Function Compute endpoint URL.
    const backendUrl = 'YOUR_ALIBABA_CLOUD_FUNCTION_URL';

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        toast({
          title: language === 'ar' ? "تم ارسال طلبك" : "Message sent successfully!",
          description: language === 'ar' ? "سيتم التواصل معكم خلال 24 ساعة" : "We'll get back to you within 24 hours.",
        });
        setFormData({
          firstName: "",
          lastName: "",
          email: "",
          phonenumber: "",
          company: "",
          companySize: "",
          message: ""
        });
      } else {
        const errorData = await response.json();
        throw new Error(errorData.message || language === 'ar' ? "حدث خطأ 504" : 'Something went wrong.');
      }
    } catch (error: any) {
      console.error( language === 'ar' ? "حدث حطأ 505" : 'Submission error:', error);
      toast({
        title: language === 'ar' ? "حدث حطأ" : 'Submission error:',
        description: language === 'ar' ? "حدث خطأ عند ارسال الطب، الرجاء المحاولة في وقت لاحق" : `There was an error sending your message. Please try again.`,
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const contactInfo = [
    {
      icon: Mail,
      title: language ==='ar' ? "البريد الالكتروني" : "Email",
      details: "info@gmai.sa",
      description: language ==='ar' ? "تواصل بأي وقت" : "Send us an email anytime",
      action: "mailto:info@gmai.sa"
    },
    {
      icon: Phone,
      title: language ==='ar' ? "الهاتف" :"Phone",
      details: language ==='ar' ? "0544669879" : "+966 (54) 466 9879",
      description: language ==='ar' ? "الأحد حتى الخميس 8:00 ص حتى 6:00 م" : "Sun-Thu from 8am to 6pm",
      action: "tel:+966544669879"
    },
    {
      icon: MessageCircle,
      title: "Whatsapp",
      details: language ==='ar' ? "0544669879" : "+966 (54) 466 9879",
      description: language ==='ar' ? "الأحد حتى الخميس 8:00 ص حتى 6:00 م" : "Sun-Thu from 8am to 6pm",
      action: "https://wa.me/966544669879"
    },
    {
      icon: MapPin,
      title: language ==='ar' ? "الموقع" : "Office",
      details: language ==='ar' ? "الرياض، المملكة العربية السعودية" :"Riyadh, Saudi Arabia",
      description: language ==='ar' ? "طريق الامير بندر بن عبدالعزيز، حي الاندلس" : "Bander Bin Abdulaziz, Al-Andalus Dist",
      action: "https://maps.app.goo.gl/NzTLx7qrRhmVCF2N9"
    }
  ];
  
  return (
    <section id="contact" className="py-20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {getTranslation("contactTitle", language)}
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            {getTranslation("contactFormDesc", language)}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Contact Information */}
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-6">
                {getTranslation("contactFormTitle", language)}</h3>
              <p className="text-muted-foreground mb-8">
                {getTranslation("contactInfo", language)}
              </p>
            </div>

            {contactInfo.map((info, index) => (
              <a href={info.action} key={index}>
                <Card className="group p-3 hover:shadow-elegant transition-all duration-300 hover:-translate-y-1 ">
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                      <info.icon className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">{info.title}</h4>
                      <p className="text-primary font-medium">{info.details}</p>
                      <p className="text-sm text-muted-foreground">{info.description}</p>
                    </div>
                  </div>
                </Card>
              </a>
            ))}
          </div>

          {/* Contact Form */}
          <div className="lg:col-span-2">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="text-2xl">
                  {getTranslation("formTitle", language)}</CardTitle>
                <CardDescription>
                  {getTranslation("formDesc", language)}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="firstName">
                        {getTranslation("firstName", language)} *</Label>
                      <Input
                        id="firstName"
                        required
                        value={formData.firstName}
                        onChange={(e) => handleInputChange("firstName", e.target.value)}
                        placeholder={language === 'ar' ? "ادخل اسمك الاول" : "Enter your first name"}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="lastName">{getTranslation("lastName", language)}</Label>
                      <Input
                        id="lastName"
                        value={formData.lastName}
                        onChange={(e) => handleInputChange("lastName", e.target.value)}
                        placeholder={language === 'ar' ? "ادخل اسمك الاخير" : "Enter your last name"}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="email">{getTranslation("email", language)} *</Label>
                      <Input
                        id="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={(e) => handleInputChange("email", e.target.value)}
                        placeholder= {language === 'ar' ? "ادخل بريدك الالكتروني" : "Enter your email address"}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phonenumber">{getTranslation("phoneNum", language)}</Label>
                      <Input
                        id="phonenumber"
                        type="tel"
                        value={formData.phonenumber}
                        onChange={(e) => handleInputChange("phonenumber", e.target.value)}
                        placeholder={language === 'ar' ? "ادخل رقم الجوال" : "Enter your phone number"}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="company">{getTranslation("companyForm", language)} *</Label>
                      <Input
                        id="company"
                        required
                        value={formData.company}
                        onChange={(e) => handleInputChange("company", e.target.value)}
                        placeholder={language === 'ar' ? "ادخل اسم الشركة" : "Enter your company name"}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="companySize">{getTranslation("companySize", language)}</Label>
                      <Select value={formData.companySize} onValueChange={(value) => handleInputChange("companySize", value)}>
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
                    <Label htmlFor="message">{getTranslation("message", language)}</Label>
                    <Textarea
                      id="message"
                      value={formData.message}
                      onChange={(e) => handleInputChange("message", e.target.value)}
                      placeholder={language === 'ar' ? "أخبرنا عنك" : "Tell us about your customer service challenges and goals..."}
                      rows={4}
                    />
                  </div>

                  <Button type="submit" variant="cta" size="lg" className="w-full group" disabled={isSubmitting}>
                    <Send className={`mr-2 h-5 w-5 ${isSubmitting ? 'animate-pulse' : ''}`} />
                    {isSubmitting ? "Sending... | جار الارسال" : "Send Message | ارسل الطلب"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
