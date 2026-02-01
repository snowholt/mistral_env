import React, { createContext, useContext, useState, useEffect } from 'react';

type Language = 'en' | 'ar';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  dir: 'ltr' | 'rtl';
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navbar
    'nav.features': 'Features',
    'nav.products': 'Products',
    'nav.about': 'About',
    'nav.contact': 'Contact',
    'nav.getStarted': 'Get Started',
    
    // Hero
    'hero.title': 'Unlock the Power of',
    'hero.titleHighlight': 'Artificial Intelligence',
    'hero.subtitle': 'Transform your business with cutting-edge AI solutions. We deliver intelligent automation, predictive analytics, and machine learning that drives real results.',
    'hero.cta': 'Start Your AI Journey',
    'hero.explore': 'Explore Solutions',
    'hero.enterprise': 'Enterprise Ready',
    'hero.uptime': '99.9% Uptime',
    'hero.support': '24/7 Support',
    
    // Features
    'features.title': 'Powerful AI',
    'features.titleHighlight': 'Capabilities',
    'features.subtitle': 'Comprehensive suite of AI-powered tools designed to revolutionize your business operations',
    'features.ml.title': 'Smart Automation',
    'features.ml.desc': 'Automate routine customer service tasks while seamlessly escalating complex issues to human agents.',
    'features.nlp.title': 'Intelligent Chat Support',
    'features.nlp.desc': 'Advanced text analysis, sentiment detection, and conversational AI that understands context and nuance.',
    'features.vision.title': 'Analytics & Insights',
    'features.vision.desc': 'Comprehensive analytics dashboard to track performance, customer satisfaction, and identify improvements.',
    'features.analytics.title': '24/7 Availability',
    'features.analytics.desc': 'Round-the-clock customer support ensuring your customers get help whenever they need it.',
    'features.automation.title': 'Rapid Integration',
    'features.automation.desc': 'Intelligent workflow automation that reduces manual tasks and accelerates your business processes.',
    'features.security.title': 'Enterprise Security',
    'features.security.desc': 'Quick and easy integration with your existing systems and platforms with minimal setup time.',
    
    // Products
    'products.title': 'Our AI',
    'products.titleHighlight': 'Products',
    'products.subtitle': 'Transform your customer service experience with our intelligent AI agents that work 24/7',
    'products.voice.title': 'S.I.N.A (Smart Interactive Natural Agent) Voice Agent',
    'products.voice.desc': 'AI voice agent that replaces call center staff, speaking naturally in Saudi dialect to answer calls, resolve inquiries, and book appointments in real-time.',
    'products.voice.tag': 'Voice AI',
    'products.whatsapp.title': 'S.I.N.A (Smart Interactive Natural Agent) WhatsApp',
    'products.whatsapp.desc': 'WhatsApp AI agent that automates conversations, ensuring instant responses, personalized interactions, and increased customer engagement.',
    'products.whatsapp.tag': 'WhatsApp AI',
    'products.chatbot.title': 'Sina Chatbot',
    'products.chatbot.desc': 'Intelligent chatbot providing instant, smart responses to website visitors, enhancing engagement and reducing frustration with fast automated support.',
    'products.chatbot.tag': 'Chat AI',
    'products.stat1': 'Wait Time Reduction',
    'products.stat2': 'Availability',
    'products.stat3': 'Cost Reduction',
    'products.stat4': 'Response Time',
    
    // About
    'about.title': 'About',
    'about.titleHighlight': 'Genius AI',
    'about.subtitle': 'Pioneering the future of artificial intelligence in Saudi Arabia',
    'about.mission': 'Our Mission',
    'about.missionText': 'At Genius AI, we\'re committed to democratizing artificial intelligence for businesses across Saudi Arabia and the MENA region. Our team of expert data scientists, engineers, and strategists work together to deliver AI solutions that create measurable impact.',
    'about.certified': 'Saudi Certified',
    'about.certifiedText': 'Fully compliant with Saudi regulations',
    'about.vision': 'Vision 2030 Aligned',
    'about.visionText': 'Supporting digital transformation',
    'about.expertise': 'Local Expertise',
    'about.expertiseText': 'Deep understanding of regional needs',
    'about.title1': "Innovation",
    'about.stat1': "Pushing the boundaries of what's possible with AI-driven communication.",
    'about.title2': "Customer-First",
    'about.stat2': 'Designing solutions that prioritize the end-user experience above all else.',
    'about.title3': "Continuous Learning",
    'about.stat3': 'Our AI agents evolve and improve with every interaction.',
    'about.title4': "Ethical AI",
    'about.stat4': 'Building reliable, transparent, and responsible AI systems.',
    
    // Contact
    'contact.title': 'Ready to',
    'contact.titleHighlight': 'Get Started?',
    'contact.subtitle': 'Let\'s discuss how Genius AI can transform your business with intelligent solutions',
    'contact.email': 'Email Us',
    'contact.call': 'Call Us',
    'contact.whatsapp': 'WhatsApp',
    'contact.visit': 'Visit Us',
    'contact.location': 'Riyadh, Prince Bandar bin Abdulaziz Road, Al Andalus District',
    'contact.hours': 'Sun - Thu 8:00 AM - 6:00 PM',
    'contact.openMap': 'Click to open in Google Maps',
    'contact.form.title': 'Request a Demo',
    'contact.form.firstName': 'First Name',
    'contact.form.lastName': 'Last Name',
    'contact.form.email': 'Email Address',
    'contact.form.phone': 'Phone Number',
    'contact.form.company': 'Company Name',
    'contact.form.companySize': 'Company Size',
    'contact.form.selectSize': 'Select company size',
    'contact.form.size1': '1-10 employees',
    'contact.form.size2': '11-50 employees',
    'contact.form.size3': '51-200 employees',
    'contact.form.size4': '201-1000 employees',
    'contact.form.size5': '1000+ employees',
    'contact.form.message': 'Your Message',
    'contact.form.submit': 'Send Message',
    'contact.form.success': 'Message sent successfully! We\'ll get back to you within 24 hours.',
    'contact.cta.title': 'Start Your Free Consultation',
    'contact.cta.text': 'Book a 30-minute call with our AI specialists to explore how we can accelerate your digital transformation.',
    'contact.cta.button': 'Schedule a Call',
    
    // Footer
    'footer.privacy': 'Privacy Policy',
    'footer.terms': 'Terms of Service',
    'footer.rights': '© 2025 Genius AI. All rights reserved.',
    
    // Privacy & Terms
    'privacy.title': 'Privacy Policy',
    'terms.title': 'Terms of Service',
    'back.home': 'Back to Home',
    
    //Error
    'error.title': 'Oops! Page not found',
    'error.content': 'Return to Home',

  },
  ar: {
    // Navbar
    'nav.features': 'المميزات',
    'nav.products': 'منتجاتنا',
    'nav.about': 'عن الشركة',
    'nav.contact': 'تواصل معنا',
    'nav.getStarted': 'ابدأ الآن',
    
    // Hero
    'hero.title': 'اطلق العنان لقوة',
    'hero.titleHighlight': 'الذكاء الاصطناعي',
    'hero.subtitle': 'حوّل عملك بحلول الذكاء الاصطناعي المتطورة. نقدم الأتمتة الذكية والتحليلات التنبؤية والتعلم الآلي الذي يحقق نتائج حقيقية.',
    'hero.cta': 'ابدأ رحلتك مع الذكاء الاصطناعي',
    'hero.explore': 'استكشف الحلول',
    'hero.enterprise': 'جاهز للمؤسسات',
    'hero.uptime': '99.9% وقت التشغيل',
    'hero.support': 'دعم على مدار الساعة',
    
    // Features
    'features.title': 'قدرات الذكاء الاصطناعي',
    'features.titleHighlight': 'المتقدمة',
    'features.subtitle': 'مجموعة شاملة من أدوات الذكاء الاصطناعي المصممة لإحداث ثورة في عمليات عملك',
    'features.ml.title': 'الأتمتة الذكية',
    'features.ml.desc': 'أتمتة مهام خدمة العملاء الروتينية مع تصعيد القضايا المعقدة بسلاسة إلى الوكلاء البشريين.',
    'features.nlp.title': 'دعم الدردشة الذكية',
    'features.nlp.desc': 'وكلاء صوتيون يعملون بالذكاء الاصطناعي ليحلوا محل موظفي مراكز الاتصال البشرية، ويتحدثون بشكل طبيعي بلهجة سعودية للرد على المكالمات وحل الاستفسارات وحجز المواعيد في الوقت الفعلي.',
    'features.vision.title': 'التحليلات والرؤى',
    'features.vision.desc': 'لوحة تحليلات شاملة لتتبع الأداء ورضا العملاء وتحديد التحسينات.',
    'features.analytics.title': 'التوفر على مدار الساعة',
    'features.analytics.desc': 'دعم العملاء على مدار الساعة لضمان حصول عملائك على المساعدة عندما يحتاجونها.',
    'features.automation.title': 'التكامل السريع',
    'features.automation.desc': 'تكامل سريع وسهل مع أنظمتك ومنصاتك الحالية مع الحد الأدنى من وقت الإعداد.',
    'features.security.title': 'أمان المؤسسات',
    'features.security.desc': 'بروتوكولات أمنية على مستوى البنوك لحماية بيانات العملاء وضمان الامتثال للوائح.',
    
    // Products
    'products.title': 'منتجات',
    'products.titleHighlight': 'الذكاء الاصطناعي',
    'products.subtitle': 'حوّل تجربة خدمة العملاء لديك مع عملاء الذكاء الاصطناعي الذين يعملون على مدار الساعة',
    'products.voice.title': 'سِنا الصوتي',
    'products.voice.desc': 'وكيل صوتي يعمل بالذكاء الاصطناعي ليحل محل موظفي مراكز الاتصال، يتحدث بشكل طبيعي باللهجة السعودية للرد على المكالمات وحل الاستفسارات وحجز المواعيد.',
    'products.voice.tag': 'ذكاء صوتي',
    'products.whatsapp.title': 'سِنا واتساب',
    'products.whatsapp.desc': 'وكيل واتساب يعمل بالذكاء الاصطناعي لأتمتة المحادثات، يضمن الرد الفوري والتفاعل الشخصي وزيادة تفاعل العملاء.',
    'products.whatsapp.tag': 'واتساب ذكي',
    'products.chatbot.title': 'سِنا روبوت الدردشة',
    'products.chatbot.desc': 'روبوت دردشة ذكي يوفر ردود فورية وذكية لزوار الموقع، يعزز التفاعل ويقلل الإحباط مع دعم آلي سريع.',
    'products.chatbot.tag': 'دردشة ذكية',
    'products.stat1': 'تقليل وقت الانتظار',
    'products.stat2': 'التوفر',
    'products.stat3': 'تقليل التكاليف',
    'products.stat4': 'وقت الاستجابة',
    
    // About
    'about.title': 'عن',
    'about.titleHighlight': 'Genius AI',
    'about.subtitle': 'رواد مستقبل الذكاء الاصطناعي في المملكة العربية السعودية',
    'about.mission': 'مهمتنا',
    'about.missionText': 'في Genius AI، نلتزم بإتاحة الذكاء الاصطناعي للشركات في جميع أنحاء المملكة العربية السعودية ومنطقة الشرق الأوسط وشمال أفريقيا. يعمل فريقنا من علماء البيانات والمهندسين والاستراتيجيين الخبراء معاً لتقديم حلول ذكاء اصطناعي تحقق تأثيراً ملموساً.',
    'about.certified': 'معتمد سعودياً',
    'about.certifiedText': 'متوافق بالكامل مع الأنظمة السعودية',
    'about.vision': 'متوافق مع رؤية 2030',
    'about.visionText': 'دعم التحول الرقمي',
    'about.expertise': 'خبرة محلية',
    'about.expertiseText': 'فهم عميق للاحتياجات الإقليمية',
    'about.title1': 'الابتكار',
    'about.stat1': 'دفع حدود الممكن باستخدام التواصل المدعوم بالذكاء الاصطناعي.',
    'about.title2': 'العميل أولاً',
    'about.stat2': 'تصميم حلول تضع تجربة المستخدم النهائي في المقام الأول قبل كل شيء.',
    'about.title3': 'التعلم المستمر',
    'about.stat3': 'تتطور وكلاؤنا المعتمدون على الذكاء الاصطناعي ويتحسنون مع كل تفاعل.',
    'about.title4': 'الذكاء الاصطناعي الأخلاقي',
    'about.stat4': 'بناء أنظمة ذكاء اصطناعي موثوقة وشفافة ومسؤولة.',
    
    // Contact
    'contact.title': 'هل أنت مستعد',
    'contact.titleHighlight': 'للبدء؟',
    'contact.subtitle': 'دعنا نناقش كيف يمكن Genius AI تحويل عملك بحلول ذكية',
    'contact.email': 'راسلنا',
    'contact.call': 'اتصل بنا',
    'contact.whatsapp': 'واتساب',
    'contact.visit': 'زرنا',
    'contact.location': 'الرياض، طريق الأمير بندر بن عبدالعزيز، حي الأندلس',
    'contact.hours': 'الأحد - الخميس 8:00 ص - 6:00 م',
    'contact.openMap': 'انقر لفتح في خرائط جوجل',
    'contact.form.title': 'طلب تجربة',
    'contact.form.firstName': 'الاسم الأول',
    'contact.form.lastName': 'الاسم الأخير',
    'contact.form.email': 'البريد الإلكتروني',
    'contact.form.phone': 'رقم الجوال',
    'contact.form.company': 'اسم الشركة',
    'contact.form.companySize': 'حجم الشركة',
    'contact.form.selectSize': 'اختر حجم الشركة',
    'contact.form.size1': '1-10 موظف',
    'contact.form.size2': '11-50 موظف',
    'contact.form.size3': '51-200 موظف',
    'contact.form.size4': '201-1000 موظف',
    'contact.form.size5': '1000+ موظف',
    'contact.form.message': 'رسالتك',
    'contact.form.submit': 'إرسال الرسالة',
    'contact.form.success': 'تم إرسال الرسالة بنجاح! سنتواصل معك خلال 24 ساعة.',
    'contact.cta.title': 'ابدأ استشارتك المجانية',
    'contact.cta.text': 'احجز مكالمة مدتها 30 دقيقة مع متخصصي الذكاء الاصطناعي لدينا لاستكشاف كيف يمكننا تسريع تحولك الرقمي.',
    'contact.cta.button': 'جدولة مكالمة',
    
    // Footer
    'footer.privacy': 'سياسة الخصوصية',
    'footer.terms': 'شروط الخدمة',
    'footer.rights': '© 2025 عبقرية الآلة لتقنية المعلومات. س.ت: 1009141819. جميع الحقوق محفوظة.',
    
    // Privacy & Terms
    'privacy.title': 'سياسة الخصوصية',
    'terms.title': 'شروط الخدمة',
    'back.home': 'العودة للرئيسية',
    
    //Error
    'error.title': 'خطأ! الصفحة غير موجودة',
    'error.content': 'العودة للصفحة الرئيسية',
    
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(() => {
    const saved = localStorage.getItem('language');
    return (saved as Language) || 'en';
  });

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('language', lang);
  };

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  const dir = language === 'ar' ? 'rtl' : 'ltr';

  useEffect(() => {
    document.documentElement.dir = dir;
    document.documentElement.lang = language;
  }, [language, dir]);

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t, dir }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
