type TranslationKey = 
  | 'home' | 'services' | 'caseStudies' | 'about' | 'contact'
  | 'signIn' | 'requestDemo' | 'language'
  | 'heroTitle' | 'heroSubtitle' | 'heroDescription' | 'requestDemoBtn' | 'watchVideo'
  | 'waitingTime' | 'availability' | 'costReduction'
  | 'servicesTitle' | 'servicesSubtitle'
  | 'intelligentChatTitle' | 'intelligentChatDesc' | 'naturalLanguage' | 'multiLanguage' | 'contextualConversations'
  | 'smartAutomationTitle' | 'smartAutomationDesc' | 'workflowAutomation' | 'smartRouting' | 'humanHandoff'
  | 'analyticsTitle' | 'analyticsDesc' | 'realTimeMetrics' | 'performanceInsights' | 'customReporting'
  | 'availabilityTitle' | 'availabilityDesc' | 'alwaysOnline' | 'globalTimezone' | 'instantResponses'
  | 'securityTitle' | 'securityDesc' | 'dataEncryption' | 'gdprCompliant' | 'socCertified'
  | 'integrationTitle' | 'integrationDesc' | 'apiIntegration' | 'pluginSupport' | 'easyDeployment'
  | 'caseStudiesTitle' | 'caseStudiesSubtitle' | 'challenge' | 'solution' | 'responseTime' | 'costSavings'
  | 'cartConversion' | 'supportTickets' | 'revenueImpact' | 'viewMoreCaseStudies'
  | 'aboutTitle' | 'aboutDescription' | 'companiesServed' | 'conversationsHandled' | 'uptimeGuarantee' | 'teamMembers'
  | 'ourValues' | 'customerCentric' | 'customerCentricDesc' | 'innovationFirst' | 'innovationFirstDesc'
  | 'partnership' | 'partnershipDesc' | 'excellence' | 'excellenceDesc' | 'leadershipTeam'
  | 'contactTitle' | 'contactFormTitle' | 'contactFormDesc' | 'getInTouch' | 'contactInfo'
  | 'name' | 'email' | 'message' | 'sendMessage' | 'phone' | 'address'
  | 'newsletterTitle' | 'newsletterDesc' | 'subscribe' | 'enterEmail' | 'companyInfo'
  | 'product' | 'company' | 'resources' | 'legal' | 'features' | 'integrations' | 'pricing'
  | 'careers' | 'press' | 'blog' | 'helpCenter' | 'community' | 'privacyPolicy' | 'termsOfService'
  | 'cookiePolicy' | 'security' | 'copyright' | 'builtWith';

const translations: Record<string, Record<TranslationKey, string>> = {
  en: {
    home: 'Home',
    services: 'Services',
    caseStudies: 'Case Studies',
    about: 'About',
    contact: 'Contact',
    signIn: 'Sign In',
    requestDemo: 'Request Demo',
    language: 'العربية',
    
    // Hero section
    heroTitle: 'Transform Customer Service with AI Agents',
    heroSubtitle: 'Intelligent AI agents that understand, engage, and resolve customer inquiries 24/7',
    heroDescription: 'Revolutionize your customer support with intelligent AI agents that provide instant, accurate, and personalized responses 24/7. Increase satisfaction while reducing costs.',
    requestDemoBtn: 'Request Demo',
    watchVideo: 'Watch Video',
    waitingTime: 'Waiting Time Reduction',
    availability: 'Availability',
    costReduction: 'Cost Reduction',
    
    // Services section
    servicesTitle: 'Comprehensive AI Customer Service Solutions',
    servicesSubtitle: 'Our AI agents are designed to handle complex customer interactions across multiple channels, providing consistent and personalized support that scales with your business.',
    intelligentChatTitle: 'Intelligent Chat Support',
    intelligentChatDesc: 'AI-powered voice agents that replace human call centre staff, speaking naturally in Saudi dialect to answer calls, resolve inquiries, and book appointments in real time.',
    naturalLanguage: 'Human-like Saudi Conversations',
    multiLanguage: 'Smart Appointment Booking',
    contextualConversations: 'Ecosystem Integration',
    smartAutomationTitle: 'Smart Automation',
    smartAutomationDesc: 'Automate routine customer service tasks while seamlessly escalating complex issues to human agents.',
    workflowAutomation: 'Workflow automation',
    smartRouting: 'Smart routing',
    humanHandoff: 'Human handoff',
    analyticsTitle: 'Analytics & Insights',
    analyticsDesc: 'Comprehensive analytics dashboard to track performance, customer satisfaction, and identify improvements.',
    realTimeMetrics: 'Real-time metrics',
    performanceInsights: 'Performance insights',
    customReporting: 'Custom reporting',
    availabilityTitle: '24/7 Availability',
    availabilityDesc: 'Round-the-clock customer support ensuring your customers get help whenever they need it.',
    alwaysOnline: 'Always online',
    globalTimezone: 'Global timezone support',
    instantResponses: 'Instant responses',
    securityTitle: 'Enterprise Security',
    securityDesc: 'Bank-level security protocols to protect customer data and ensure compliance with regulations.',
    dataEncryption: 'Data encryption',
    gdprCompliant: 'GDPR compliant',
    socCertified: 'SOC 2 certified',
    integrationTitle: 'Rapid Integration',
    integrationDesc: 'Quick and easy integration with your existing systems and platforms with minimal setup time.',
    apiIntegration: 'API integration',
    pluginSupport: 'Plugin support',
    easyDeployment: 'Easy deployment',
    
    // Case Studies section
    caseStudiesTitle: 'Real Results from Real Companies',
    caseStudiesSubtitle: 'See how leading companies have transformed their customer service with our AI agents',
    challenge: 'Challenge',
    solution: 'Solution',
    responseTime: 'Response Time',
    costSavings: 'Cost Savings',
    cartConversion: 'Cart Conversion',
    supportTickets: 'Support Tickets',
    revenueImpact: 'Revenue Impact',
    viewMoreCaseStudies: 'View More Case Studies',
    
    // About section
    aboutTitle: 'About Genius AI',
    aboutDescription: "We're revolutionizing customer service through intelligent AI agents that understand, learn, and adapt to provide exceptional customer experiences. Founded in 2024, we've helped hundreds of companies transform their customer support operations with cutting-edge AI technology.",
    companiesServed: 'Companies Served',
    conversationsHandled: 'Conversations Handled',
    uptimeGuarantee: 'Uptime Guarantee',
    teamMembers: 'Team Members',
    ourValues: 'Our Values',
    customerCentric: 'Customer-Centric',
    customerCentricDesc: 'Everything we build is designed to enhance the customer experience and drive satisfaction.',
    innovationFirst: 'Innovation First',
    innovationFirstDesc: 'We leverage cutting-edge AI technology to solve complex customer service challenges.',
    partnership: 'Partnership',
    partnershipDesc: 'We work closely with our clients to understand their unique needs and deliver tailored solutions.',
    excellence: 'Excellence',
    excellenceDesc: 'We maintain the highest standards in AI development, security, and customer support.',
    leadershipTeam: 'Leadership Team',
    
    // Contact section
    contactTitle: 'Contact Us',
    contactFormTitle: 'Get In Touch',
    contactFormDesc: 'Ready to transform your customer service? Contact us today for a personalized demo.',
    getInTouch: 'Get In Touch',
    contactInfo: 'Contact Information',
    name: 'Name',
    email: 'Email',
    message: 'Message',
    sendMessage: 'Send Message',
    phone: 'Phone',
    address: 'Address',
    
    // Footer section
    newsletterTitle: 'Stay Updated with AI Customer Service Insights',
    newsletterDesc: 'Get the latest trends, tips, and updates delivered to your inbox monthly.',
    subscribe: 'Subscribe',
    enterEmail: 'Enter your email',
    companyInfo: 'Transforming customer service with intelligent AI agents that provide exceptional experiences 24/7. Join hundreds of companies already using our platform.',
    product: 'Product',
    company: 'Company',
    resources: 'Resources',
    legal: 'Legal',
    features: 'Features',
    integrations: 'Integrations',
    pricing: 'Pricing',
    careers: 'Careers',
    press: 'Press',
    blog: 'Blog',
    helpCenter: 'Help Center',
    community: 'Community',
    privacyPolicy: 'Privacy Policy',
    termsOfService: 'Terms of Service',
    cookiePolicy: 'Cookie Policy',
    security: 'Security',
    copyright: '© 2025 Genius AI. All rights reserved.',
    builtWith: 'Built with ❤️ for better customer experiences'
  },
  ar: {
    home: 'الرئيسية',
    services: 'الخدمات',
    caseStudies: 'دراسات الحالة',
    about: 'نبذة عنا',
    contact: 'اتصل بنا',
    signIn: 'تسجيل الدخول',
    requestDemo: 'طلب عرض تجريبي',
    language: 'English',
    
    // Hero section
    heroTitle: 'غيّر تجربة خدمة العملاء بواسطة عملاء الذكاء الاصطناعي',
    heroSubtitle: 'عملاء الذكاء الاصطناعي قادرون على الفهم والتفاعل وحل استفسارات العملاء على مدار الساعة',
    heroDescription: 'نحدث نقلة نوعية في تجربة دعم العملاء باستخدام الذكاء الاصطناعي لينوب عن موظف خدمة العملاء . برنامجنا يقدم استجابات فورية على مدار الساعة طوال أيام الأسبوع، مع إجابات دقيقة وشخصية تناسب كل عميل. مما يمكن ذلك خفض التكاليف على مركز خدمة العملاء',
    requestDemoBtn: 'طلب عرض تجريبي',
    watchVideo: 'مشاهدة الفيديو',
    waitingTime: 'تقليل وقت الانتظار',
    availability: 'التوفر',
    costReduction: 'تكلفة أقل',
    
    // Services section
    servicesTitle: 'حلول شاملة لخدمة العملاء بالذكاء الاصطناعي',
    servicesSubtitle: 'تم تصميم عميل الذكاء الاصطناعي لدينا للتعامل مع تفاعلات العملاء المعقدة عبر قنوات متعددة، وتوفير دعم متسق وشخصي يتوسع مع عملك.',
    intelligentChatTitle: 'دعم الدردشة الذكية',
    intelligentChatDesc:  'وكلاء صوتيون يعملون بالذكاء الاصطناعي ليحلوا محل موظفي مراكز الاتصال البشرية، ويتحدثون بشكل طبيعي بلهجة سعودية للرد على المكالمات وحل الاستفسارات وحجز المواعيد في الوقت الفعلي.',
    naturalLanguage: 'محادثات بلهجة سعودية شبيهة بالبشر',
    multiLanguage: 'حجز المواعيد الذكي',
    contextualConversations: 'تكامل الأنظمة',
    smartAutomationTitle: 'الأتمتة الذكية',
    smartAutomationDesc: 'أتمتة مهام خدمة العملاء الروتينية مع تصعيد القضايا المعقدة بسلاسة إلى الوكلاء البشريين.',
    workflowAutomation: 'أتمتة سير العمل',
    smartRouting: 'التوجيه الذكي',
    humanHandoff: 'تحويل إلى موظف بشري',
    analyticsTitle: 'التحليلات والرؤى',
    analyticsDesc: 'لوحة تحليلات شاملة لتتبع الأداء ورضا العملاء وتحديد التحسينات.',
    realTimeMetrics: 'مقاييس في الوقت الفعلي',
    performanceInsights: 'رؤى الأداء',
    customReporting: 'تقارير مخصصة',
    availabilityTitle: 'التوفر على مدار الساعة',
    availabilityDesc: 'دعم العملاء على مدار الساعة لضمان حصول عملائك على المساعدة عندما يحتاجونها.',
    alwaysOnline: 'متاح دائماً',
    globalTimezone: 'دعم المناطق الزمنية العالمية',
    instantResponses: 'استجابات فورية',
    securityTitle: 'أمان المؤسسات',
    securityDesc: 'بروتوكولات أمنية على مستوى البنوك لحماية بيانات العملاء وضمان الامتثال للوائح.',
    dataEncryption: 'تشفير البيانات',
    gdprCompliant: 'متوافق مع اللائحة العامة لحماية البيانات',
    socCertified: 'معتمد SOC 2',
    integrationTitle: 'التكامل السريع',
    integrationDesc: 'تكامل سريع وسهل مع أنظمتك ومنصاتك الحالية مع الحد الأدنى من وقت الإعداد.',
    apiIntegration: 'تكامل API',
    pluginSupport: 'دعم الإضافات',
    easyDeployment: 'نشر سهل',
    
    // Case Studies section
    caseStudiesTitle: 'نتائج حقيقية من شركات حقيقية',
    caseStudiesSubtitle: 'اطلع على كيفية تحويل الشركات الرائدة لخدمة العملاء باستخدام وكلاء الذكاء الاصطناعي لدينا',
    challenge: 'التحدي',
    solution: 'الحل',
    responseTime: 'وقت الاستجابة',
    costSavings: 'تكلفة أقل',
    cartConversion: 'تحويل السلة',
    supportTickets: 'تذاكر الدعم',
    revenueImpact: 'تأثير على الإيرادات',
    viewMoreCaseStudies: 'عرض المزيد من دراسات الحالة',
    
    // About section
    aboutTitle: 'نبذة عن عبقرية الآلة',
    aboutDescription: 'نحن نحدث ثورة في خدمة العملاء من خلال وكلاء ذكاء اصطناعي أذكياء يفهمون ويتعلمون ويتكيفون لتقديم تجارب استثنائية للعملاء. منذ تأسيسنا في عام 2024، ساعدنا مئات الشركات على تغيير عمليات دعم العملاء الخاصة بهم باستخدام أحدث تقنيات الذكاء الاصطناعي.',
    companiesServed: 'شركات تم خدمتها',
    conversationsHandled: 'محادثات تمت معالجتها',
    uptimeGuarantee: 'ضمان التوفر',
    teamMembers: 'أعضاء الفريق',
    ourValues: 'قيمنا',
    customerCentric: 'التركيز على العميل',
    customerCentricDesc: 'كل ما نبنيه مصمم لتحسين تجربة العميل وتحقيق رضاه.',
    innovationFirst: 'الابتكار أولاً',
    innovationFirstDesc: 'نحن نستفيد من تقنية الذكاء الاصطناعي المتطورة لحل تحديات خدمة العملاء المعقدة.',
    partnership: 'الشراكة',
    partnershipDesc: 'نحن نعمل بشكل وثيق مع عملائنا لفهم احتياجاتهم الفريدة وتقديم حلول مخصصة.',
    excellence: 'التميز',
    excellenceDesc: 'نحن نحافظ على أعلى المعايير في تطوير الذكاء الاصطناعي والأمان ودعم العملاء.',
    leadershipTeam: 'فريق القيادة',
    
    // Contact section
    contactTitle: 'اتصل بنا',
    contactFormTitle: 'تواصل معنا',
    contactFormDesc: 'هل أنت مستعد لتغيير خدمة عملائك؟ اتصل بنا اليوم للحصول على عرض تجريبي مخصص.',
    getInTouch: 'تواصل معنا',
    contactInfo: 'معلومات الاتصال',
    name: 'الاسم',
    email: 'البريد الإلكتروني',
    message: 'الرسالة',
    sendMessage: 'إرسال الرسالة',
    phone: 'الهاتف',
    address: 'العنوان',
    
    // Footer section
    newsletterTitle: 'ابق على اطلاع بأحدث رؤى خدمة العملاء بالذكاء الاصطناعي',
    newsletterDesc: 'احصل على أحدث الاتجاهات والنصائح والتحديثات المرسلة إلى بريدك الوارد شهرياً.',
    subscribe: 'اشترك',
    enterEmail: 'أدخل بريدك الإلكتروني',
    companyInfo: 'تحويل خدمة العملاء بوكلاء ذكاء اصطناعي ذكيين يقدمون تجارب استثنائية على مدار الساعة. انضم إلى مئات الشركات التي تستخدم منصتنا بالفعل.',
    product: 'المنتج',
    company: 'الشركة',
    resources: 'الموارد',
    legal: 'قانوني',
    features: 'الميزات',
    integrations: 'التكاملات',
    pricing: 'التسعير',
    careers: 'الوظائف',
    press: 'الصحافة',
    blog: 'المدونة',
    helpCenter: 'مركز المساعدة',
    community: 'المجتمع',
    privacyPolicy: 'سياسة الخصوصية',
    termsOfService: 'شروط الخدمة',
    cookiePolicy: 'سياسة ملفات تعريف الارتباط',
    security: 'الأمان',
    copyright: '© 2025 Genius AI. جميع الحقوق محفوظة.',
    builtWith: 'مصنوع بـ ❤️ لتجارب عملاء أفضل'
  }
};

export const getTranslation = (key: TranslationKey, language: string): string => {
  return translations[language]?.[key] || translations.en[key];
};