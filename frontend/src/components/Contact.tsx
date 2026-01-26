import { useState } from "react";
import { Mail, MapPin, Phone, Send, CheckCircle, MessageCircle } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useToast } from "@/hooks/use-toast";

const PHONE_NUMBER = "+966544669879";
const WHATSAPP_LINK = "https://wa.me/966544669879";
const GOOGLE_MAPS_LINK = "https://maps.app.goo.gl/NzTLx7qrRhmVCF2N9";

const Contact = () => {
  const { t } = useLanguage();
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    company: '',
    companySize: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const companySizes = [
    { value: '1-10', label: t('contact.form.size1') },
    { value: '11-50', label: t('contact.form.size2') },
    { value: '51-200', label: t('contact.form.size3') },
    { value: '201-1000', label: t('contact.form.size4') },
    { value: '1000+', label: t('contact.form.size5') },
  ];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    const fullName = `${formData.firstName} ${formData.lastName}`.trim();
    const subject = encodeURIComponent(`Contact from ${fullName} - ${formData.company}`);
    const body = encodeURIComponent(
      `Name: ${fullName}\nEmail: ${formData.email}\nPhone: ${formData.phone}\nCompany: ${formData.company}\nCompany Size: ${formData.companySize}\n\nMessage:\n${formData.message}`
    );

    window.location.href = `mailto:info@gmai.sa?subject=${subject}&body=${body}`;

    setTimeout(() => {
      setIsSubmitting(false);
      setIsSubmitted(true);
      toast({
        title: "Email Client Opened",
        description: t('contact.form.success'),
      });

      setTimeout(() => {
        setIsSubmitted(false);
        setFormData({ firstName: '', lastName: '', email: '', phone: '', company: '', companySize: '', message: '' });
      }, 3000);
    }, 500);
  };

  return (
    <section id="contact" className="py-24 relative">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

      <div className="container mx-auto px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-display font-bold mb-4">
              {t('contact.title')} <span className="text-gradient">{t('contact.titleHighlight')}</span>
            </h2>
            <p className="text-muted-foreground text-lg max-w-xl mx-auto">
              {t('contact.subtitle')}
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12">
            {/* Contact Info Cards */}
            <div className="space-y-6">
              <div className="grid gap-6">
                {/* Email */}
                <a
                  href="mailto:info@gmai.sa"
                  className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-300"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                      <Mail className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-display font-semibold mb-1">{t('contact.email')}</h3>
                      <p className="text-primary">info@gmai.sa</p>
                    </div>
                  </div>
                </a>

                {/* Call Us - Clickable */}
                <a
                  href={`tel:${PHONE_NUMBER}`}
                  className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-300 cursor-pointer"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                      <Phone className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-display font-semibold mb-1">{t('contact.call')}</h3>
                      <p className="text-primary">0544669879</p>
                      <p className="text-muted-foreground text-sm">{t('contact.hours')}</p>
                    </div>
                  </div>
                </a>

                {/* WhatsApp */}
                <a
                  href={WHATSAPP_LINK}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-300 cursor-pointer"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                      <MessageCircle className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-display font-semibold mb-1">{t('contact.whatsapp')}</h3>
                      <p className="text-primary">0544669879</p>
                      <p className="text-muted-foreground text-sm">{t('contact.hours')}</p>
                    </div>
                  </div>
                </a>

                {/* Visit Us - Opens Google Maps */}
                <a
                  href={GOOGLE_MAPS_LINK}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-300 cursor-pointer"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                      <MapPin className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-display font-semibold mb-1">{t('contact.visit')}</h3>
                      <p className="text-muted-foreground">{t('contact.location')}</p>
                    </div>
                  </div>
                </a>
              </div>
            </div>

            {/* Contact Form */}
            <div className="bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8">
              <h3 className="text-2xl font-display font-bold mb-6">{t('contact.form.title')}</h3>

              <form onSubmit={handleSubmit} className="space-y-5">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="firstName" className="block text-sm font-medium mb-2">
                      {t('contact.form.firstName')} *
                    </label>
                    <input
                      type="text"
                      id="firstName"
                      name="firstName"
                      value={formData.firstName}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                      placeholder={t('contact.form.firstName')}
                    />
                  </div>
                  <div>
                    <label htmlFor="lastName" className="block text-sm font-medium mb-2">
                      {t('contact.form.lastName')}
                    </label>
                    <input
                      type="text"
                      id="lastName"
                      name="lastName"
                      value={formData.lastName}
                      onChange={handleChange}
                      className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                      placeholder={t('contact.form.lastName')}
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium mb-2">
                    {t('contact.form.email')} *
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                    placeholder={t('contact.form.email')}
                  />
                </div>

                <div>
                  <label htmlFor="phone" className="block text-sm font-medium mb-2">
                    {t('contact.form.phone')}
                  </label>
                  <input
                    type="tel"
                    id="phone"
                    name="phone"
                    value={formData.phone}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                    placeholder={t('contact.form.phone')}
                  />
                </div>

                <div>
                  <label htmlFor="company" className="block text-sm font-medium mb-2">
                    {t('contact.form.company')} *
                  </label>
                  <input
                    type="text"
                    id="company"
                    name="company"
                    value={formData.company}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                    placeholder={t('contact.form.company')}
                  />
                </div>

                <div>
                  <label htmlFor="companySize" className="block text-sm font-medium mb-2">
                    {t('contact.form.companySize')}
                  </label>
                  <select
                    id="companySize"
                    name="companySize"
                    value={formData.companySize}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors"
                  >
                    <option value="">{t('contact.form.selectSize')}</option>
                    {companySizes.map((size) => (
                      <option key={size.value} value={size.value}>
                        {size.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-medium mb-2">
                    {t('contact.form.message')}
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    rows={4}
                    className="w-full px-4 py-3 bg-background border border-border/50 rounded-xl focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-colors resize-none"
                    placeholder={t('contact.form.message')}
                  />
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting || isSubmitted}
                  className="w-full bg-gradient-primary text-primary-foreground px-6 py-4 rounded-xl font-semibold text-lg hover:opacity-90 transition-all duration-300 glow-primary hover:scale-[1.02] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmitted ? (
                    <>
                      <CheckCircle className="w-5 h-5" />
                      {t('contact.form.success')}
                    </>
                  ) : isSubmitting ? (
                    <span className="animate-pulse">Sending...</span>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      {t('contact.form.submit')}
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* CTA */}
          <div className="mt-16 bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 border border-primary/20 rounded-3xl p-10 text-center">
            <h3 className="text-2xl font-display font-bold mb-4">
              {t('contact.cta.title')}
            </h3>
            <p className="text-muted-foreground mb-6 max-w-lg mx-auto">
              {t('contact.cta.text')}
            </p>
            <a
              href="mailto:info@gmai.sa"
              className="inline-flex bg-gradient-primary text-primary-foreground px-8 py-4 rounded-xl font-semibold text-lg hover:opacity-90 transition-all duration-300 glow-primary hover:scale-105"
            >
              {t('contact.cta.button')}
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
