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
  | 'formTitle'| 'formDesc' | 'firstName' | 'lastName' | 'phoneNum' | 'companyForm' | 'companySize' 
  | 'name' | 'email' | 'message' | 'sendMessage' | 'phone' | 'address'|  'newsletterTitle' | 'newsletterDesc' 
  | 'subscribe' | 'enterEmail' | 'companyInfo'
  | 'product' | 'company' | 'resources' | 'legal' | 'features' | 'integrations' | 'pricing'
  | 'careers' | 'press' | 'blog' | 'helpCenter' | 'community' | 'privacyPolicy' | 'termsOfService'
  | 'cookiePolicy' | 'security' | 'copyright' | 'builtWith' | 'policyTitle' | 'policy_1_t' | 'policy_1_c' 
  | 'policy_2_t' | 'policy_2_c' | 'policy_2_c_a_t' | 'policy_2_c_a_c' | 'policy_2_c_b_t' | 'policy_2_c_b_c' 
  | 'policy_2_c_c_t' | 'policy_2_c_c_c' | 'policy_3_t' | 'policy_3_c' | 'policy_4_t' | 'policy_4_c' | 'policy_5_t' 
  | 'policy_5_c' | 'policy_6_t' | 'policy_6_c' | 'policy_7_t' | 'policy_7_c' | 'policy_8_t' | 'policy_8_c'
  | 'policy_9_t' | 'policy_9_c' | 'policy_10_t' | 'policy_10_c' | 'termTitle' | 'term_1_t' | 'term_1_c'
  | 'term_2_t' | 'term_2_c' | 'term_3_t' | 'term_3_c' | 'term_4_t' | 'term_4_c' | 'term_5_t' | 'term_5_c'
  | 'term_6_t' | 'term_6_c' | 'term_7_t' | 'term_7_c' | 'term_8_t' | 'term_8_c' | 'term_9_t' | 'term_9_c'
  | 'term_10_t' | 'term_10_c' | 'term_11_t' | 'term_11_c';

const translations: Record<string, Record<TranslationKey, string>> = {
  en: {
    home: 'Home',
    services: 'Features',
    caseStudies: 'Our Products',
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
    aboutDescription: "We're revolutionizing customer service through intelligent AI agents that understand, learn, and adapt to provide exceptional customer experiences. Founded in 2024, we've helped many of companies transform their customer support operations with cutting-edge AI technology.",
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

    // Contact form section
    formTitle: 'Request a Demo',
    formDesc: "Fill out the form below and we'll get back to you within 24 hours",
    firstName: "First Name",
    lastName: "Last Name",
    phoneNum: "Phone Number",
    companyForm: "Company",
    companySize: "Company Size",
    
    // Footer section
    newsletterTitle: 'Stay Updated with AI Customer Service Insights',
    newsletterDesc: 'Get the latest trends, tips, and updates delivered to your inbox monthly.',
    subscribe: 'Subscribe',
    enterEmail: 'Enter your email',
    companyInfo: 'Transforming customer service with intelligent AI agents that provide exceptional experiences 24/7. Join many of companies already using our platform.',
    product: 'Product',
    company: 'Browse',
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
    builtWith: 'Built with ❤️ for better customer experiences',


    // Policy Section
    policyTitle: '🔐 Privacy Policy',
    policy_1_t: '1. Introduction',
    policy_1_c: 'Genius AI (“we”, “our”, “us”) operates a messaging and communication platform that enables businesses to communicate with their customers via WhatsApp and other messaging channels. We are committed to protecting your privacy and complying with applicable data protection laws.',
    policy_2_t: '2. Information We Collect',
    policy_2_c: ' We do not read, sell, or use message content for advertising. We may collect and process the following data:',
    policy_2_c_a_t: 'a) Business Information',
    policy_2_c_a_c: 'Business name, Contact email, Phone numbers, connected to WhatsApp Business API, Business identifiers required for onboarding',
    policy_2_c_b_t: 'b) User Information',
    policy_2_c_b_c: 'Name, Email address, Role and permissions within the platform',
    policy_2_c_c_t: 'c) Messaging Data',
    policy_2_c_c_c: 'Message metadata (timestamps, delivery status), Message content only as required to deliver the service, Webhook event data from WhatsApp APIs',
    policy_3_t: '3. How We Use Information',
    policy_3_c: 'We use collected data to: Provide WhatsApp messaging services, Authenticate users via Meta OAuth, Onboard and manage WhatsApp Business Accounts, Deliver messages and receive webhooks, Improve platform performance and reliability, Comply with legal obligations',
    policy_4_t: '4. Data Sharing',
    policy_4_c: 'We may share data only with: Meta Platforms, Inc. (WhatsApp Cloud API), Authorized service providers (hosting, security, logging), Legal authorities when required by law, We do not sell or rent personal data.',
    policy_5_t: '5. Data Retention',
    policy_5_c: 'Message data is retained only as long as necessary. Logs and metadata are retained for audit and compliance. Users may request deletion of their data',
    policy_6_t: '6. Data Security',
    policy_6_c: 'We implement: Encryption in transit, Access control & authentication, Secure token handling, Regular security reviews',
    policy_7_t: '7. User Rights',
    policy_7_c: 'You have the right to: Access your data, Request correction or deletion, Withdraw consent, Request account termination requests can be sent to: info@gmai.sa',
    policy_8_t: '8. Third-Party Services',
    policy_8_c: 'Our platform integrates with Meta (Facebook / WhatsApp APIs). Use of WhatsApp is subject to Meta’s own policies.',
    policy_9_t: '9. Changes to This Policy',
    policy_9_c: 'We may update this Privacy Policy periodically. Changes will be published on this page.',
    policy_10_t: '10. Contact Us',
    policy_10_c: 'Genius AI info@gmai.sa',


    // Term Section
    termTitle: '📜 Terms of Service',
    term_1_t: '1. Acceptance of Terms',
    term_1_c: 'By accessing or using Genius AI or any of its services, you agree to these Terms of Service. If you do not agree, you must not use the service.',
    term_2_t: '2. Description of Service',
    term_2_c: 'We provide a cloud-based platform that allows businesses to: Connect WhatsApp Business accounts. Send and receive messages via WhatsApp APIs. Manage customer communications',
    term_3_t: '3. Eligibility',
    term_3_c: 'You must: Be a legally registered business. Have authority to represent your business. Comply with WhatsApp and Meta policies',
    term_4_t: '4. User Responsibilities',
    term_4_c: 'You agree to: Use the platform lawfully. Obtain customer consent before messaging. Not send spam or prohibited content. Follow WhatsApp Business Messaging Policies. You are solely responsible for the content you send.',
    term_5_t: '5. WhatsApp & Meta Compliance',
    term_5_c: 'Use of our service requires compliance with: WhatsApp Business Messaging Policy. Meta Platform Policies. Violation may result in:. Account suspension. Termination without notice',
    term_6_t: '6. Account Suspension',
    term_6_c: 'We reserve the right to suspend or terminate accounts that: Violate laws or policies. Abuse the platform. Cause reputational or technical risk',
    term_7_t: '7. Data & Privacy',
    term_7_c: 'Your use of the service is also governed by our Privacy Policy.',
    term_8_t: '8. Limitation of Liability',
    term_8_c: 'To the maximum extent permitted by law: We are not liable for indirect or consequential damages. We are not responsible for WhatsApp service outages. The service is provided “as is”',
    term_9_t: '9. Modifications',
    term_9_c: 'We may modify these terms at any time. Continued use constitutes acceptance.',
    term_10_t: '10. Governing Law',
    term_10_c: 'These terms are governed by the laws of Kingdom of Saudi Arabia (or your jurisdiction).',
    term_11_t: '11. Contact Information',
    term_11_c: 'Genius AI info@gmai.sa',
  },
  ar: {
    home: 'الرئيسية',
    services: 'المميزات',
    caseStudies: 'منتجاتنا',
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
    aboutDescription: 'نحن نحدث ثورة في خدمة العملاء من خلال وكلاء ذكاء اصطناعي أذكياء يفهمون ويتعلمون ويتكيفون لتقديم تجارب استثنائية للعملاء. منذ تأسيسنا في عام 2024، ساعدنا العديد الشركات على تغيير عمليات دعم العملاء الخاصة بهم باستخدام أحدث تقنيات الذكاء الاصطناعي.',
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

    // Contact from section
    formTitle: 'طلب تجربة',
    formDesc: 'املئ الاستبانة وسيتم التواصل معكم خلال 24 ساعة',
    firstName: "الاسم الاول",
    lastName: "الاسم الاخير",
    phoneNum: "رقم الجوال",
    companyForm: "الشركة",
    companySize: "حجم الشركة",


    // Footer section
    newsletterTitle: 'ابق على اطلاع بأحدث رؤى خدمة العملاء بالذكاء الاصطناعي',
    newsletterDesc: 'احصل على أحدث الاتجاهات والنصائح والتحديثات المرسلة إلى بريدك الوارد شهرياً.',
    subscribe: 'اشترك',
    enterEmail: 'أدخل بريدك الإلكتروني',
    companyInfo: 'تحويل خدمة العملاء بوكلاء ذكاء اصطناعي ذكيين يقدمون تجارب استثنائية على مدار الساعة. انضم إلى العديد الشركات التي تستخدم منصتنا بالفعل.',
    product: 'المنتج',
    company: 'تصفح',
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
    builtWith: 'مصنوع بـ ❤️ لتجارب عملاء أفضل',

    // Policy Section
    policyTitle: "🔐 سياسة الخصوصية",
    policy_1_t: "1. مقدمة",
    policy_1_c: "تدير شركة عبقرية الآلة للذكاء الاصطناعي (\"نحن\"، \"الشركة\") منصة مراسلة واتصال تمكّن الشركات من التواصل مع عملائها عبر واتساب وقنوات المراسلة الأخرى. نحن ملتزمون بحماية خصوصيتك والامتثال لقوانين حماية البيانات المعمول بها.",
    policy_2_t: "2. المعلومات التي نجمعها",
    policy_2_c: " نحن لا نقرأ محتوى الرسائل أو نبيعه أو نستخدمه لأغراض إعلانية. قد نقوم بجمع ومعالجة البيانات التالية:",
    policy_2_c_a_t: "أ) معلومات العمل (الشركة)",
    policy_2_c_a_c: "اسم العمل، بريد الاتصال الإلكتروني، أرقام الهواتف المتصلة بواجهة برمجة تطبيقات الأعمال في واتساب (WhatsApp Business API)، المعرّفات الخاصة بالعمل المطلوبة لإتمام عملية الإعداد",
    policy_2_c_b_t: "ب) معلومات المستخدم",
    policy_2_c_b_c: "الاسم، عنوان البريد الإلكتروني، الدور والصلاحيات ضمن المنصة",
    policy_2_c_c_t: "ج) بيانات المراسلة",
    policy_2_c_c_c: "البيانات الوصفية للرسالة (الطوابع الزمنية، حالة التسليم)، محتوى الرسالة فقط بالقدر المطلوب لتقديم الخدمة، بيانات أحداث Webhook من واجهات برمجة تطبيقات واتساب",
    policy_3_t: "3. كيف نستخدم المعلومات",
    policy_3_c: "نستخدم البيانات المجمّعة من أجل: توفير خدمات المراسلة عبر واتساب، مصادقة المستخدمين عبر Meta OAuth، إعداد وإدارة حسابات واتساب للأعمال، تسليم الرسائل واستقبال webhooks، تحسين أداء المنصة وموثوقيتها، الامتثال للالتزامات القانونية",
    policy_4_t: "4. مشاركة البيانات",
    policy_4_c: "قد نشارك البيانات فقط مع: شركة Meta Platforms, Inc. (منصة WhatsApp Cloud API)، مقدمي الخدمات المعتمدين (الاستضافة، الأمان، التسجيل)، السلطات القانونية عندما يقتضي القانون ذلك، نحن لا نبيع أو نؤجر البيانات الشخصية.",
    policy_5_t: "5. الاحتفاظ بالبيانات",
    policy_5_c: "يتم الاحتفاظ ببيانات الرسائل فقط طالما كان ذلك ضروريًا. يتم الاحتفاظ بالسجلات والبيانات الوصفية (Metadata) للمراجعة والامتثال. يجوز للمستخدمين طلب حذف بياناتهم",
    policy_6_t: "6. أمن البيانات",
    policy_6_c: "نحن ننفذ: التشفير أثناء النقل، التحكم في الوصول والمصادقة، التعامل الآمن مع الرموز المميزة (Tokens)، مراجعات أمنية منتظمة",
    policy_7_t: "7. حقوق المستخدم",
    policy_7_c: "يحق لك: الوصول إلى بياناتك، طلب التصحيح أو الحذف، سحب الموافقة، طلب إنهاء الحساب يمكن إرسال الطلبات إلى:  info@gmai.sa",
    policy_8_t: "8. خدمات الطرف الثالث",
    policy_8_c: "تتكامل منصتنا مع Meta (واجهات برمجة تطبيقات Facebook / WhatsApp). يخضع استخدام واتساب لسياسات Meta الخاصة.",
    policy_9_t: "9. التغييرات على هذه السياسة",
    policy_9_c: "قد نقوم بتحديث سياسة الخصوصية هذه بشكل دوري. سيتم نشر التغييرات على هذه الصفحة.",
    policy_10_t: "10. اتصل بنا",
    policy_10_c: "عبقرية الآلة للذكاء الاصطناعي info@gmai.sa",


    // Terms Section
    termTitle: "📜 شروط الخدمة",
    term_1_t: "1. قبول الشروط",
    term_1_c: "بمجرد الوصول إلى أو استخدام منصة عبقرية الآلة للذكاء الاصطناعي (Genius AI) أو أي من خدماتها، فإنك توافق على شروط الخدمة هذه. إذا كنت لا توافق، يجب عليك عدم استخدام الخدمة.",
    term_2_t: "2. وصف الخدمة",
    term_2_c: "نحن نوفر منصة قائمة على السحابة (Cloud-based) تتيح للشركات القيام بما يلي: ربط حسابات واتساب للأعمال. إرسال واستقبال الرسائل عبر واجهات برمجة تطبيقات واتساب (WhatsApp APIs). إدارة اتصالات العملاء.",
    term_3_t: "3. الأهلية",
    term_3_c: "يجب عليك: أن تكون عملاً تجاريًا مسجلاً قانونًا. أن تكون لديك السلطة لتمثيل عملك. الامتثال لسياسات واتساب وميتا (Meta).",
    term_4_t: "4. مسؤوليات المستخدم",
    term_4_c: "أنت توافق على: استخدام المنصة بشكل قانوني. الحصول على موافقة العميل قبل إرسال الرسائل. عدم إرسال الرسائل العشوائية (Spam) أو المحتوى المحظور. اتباع سياسات مراسلة الأعمال الخاصة بواتساب (WhatsApp Business Messaging Policies). أنت مسؤول وحدك عن المحتوى الذي ترسله.",
    term_5_t: "5. الامتثال لـ واتساب وميتا (WhatsApp & Meta)",
    term_5_c: "يتطلب استخدام خدمتنا الامتثال لما يلي: سياسة مراسلة الأعمال الخاصة بواتساب. سياسات منصة ميتا (Meta Platform Policies). قد يؤدي الانتهاك إلى: تعليق الحساب. إنهاء الخدمة دون إشعار.",
    term_6_t: "6. تعليق الحساب",
    term_6_c: "نحتفظ بالحق في تعليق أو إنهاء الحسابات التي: تنتهك القوانين أو السياسات. تسيء استخدام المنصة. تسبب خطرًا على السمعة أو خطرًا تقنيًا.",
    term_7_t: "7. البيانات والخصوصية",
    term_7_c: "يخضع استخدامك للخدمة أيضًا لسياسة الخصوصية الخاصة بنا.",
    term_8_t: "8. تحديد المسؤولية",
    term_8_c: "إلى الحد الأقصى الذي يسمح به القانون: نحن لسنا مسؤولين عن الأضرار غير المباشرة أو التبعية. نحن لسنا مسؤولين عن انقطاع خدمات واتساب. يتم توفير الخدمة \"كما هي\".",
    term_9_t: "9. التعديلات",
    term_9_c: "قد نقوم بتعديل هذه الشروط في أي وقت. استمرار استخدامك يشكل قبولاً لهذه التعديلات.",
    term_10_t: "10. القانون الحاكم",
    term_10_c: "تخضع هذه الشروط لقوانين المملكة العربية السعودية (أو ولايتك القضائية).",
    term_11_t: "11. معلومات الاتصال",
    term_11_c: "عبقرية الآلة للذكاء الاصطناعي info@gmai.sa"
  }
};

export const getTranslation = (key: TranslationKey, language: string): string => {
  return translations[language]?.[key] || translations.en[key];
};