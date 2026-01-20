import { useLanguage } from "@/hooks/useLanguage";
import RequestDemoForm from "@/components/RequestDemoForm";
import Navbar from "@/components/landing/Navbar";
import Footer from "@/components/landing/Footer";

export default function RequestDemo() {
  const { language } = useLanguage();

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <Navbar />
      
      {/* Hero Section */}
      <section className="pt-32 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            {language === 'ar' 
              ? 'جرّب BeautyAI الآن' 
              : 'Try BeautyAI Now'
            }
          </h1>
          <p className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto mb-8">
            {language === 'ar'
              ? 'احصل على وصول تجريبي مجاني لتجربة قوة المحادثات الصوتية بالذكاء الاصطناعي'
              : 'Get free demo access to experience the power of AI voice conversations'
            }
          </p>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'محادثات صوتية' : 'Voice Conversations'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar' 
                  ? 'تحدث بشكل طبيعي مع الذكاء الاصطناعي' 
                  : 'Talk naturally with AI'
                }
              </p>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'استجابة فورية' : 'Instant Response'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar' 
                  ? 'ردود سريعة في أقل من ثانيتين' 
                  : 'Fast replies in under 2 seconds'
                }
              </p>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                </svg>
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'دعم اللغة العربية' : 'Arabic Support'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar' 
                  ? 'محسّن للغة العربية والإنجليزية' 
                  : 'Optimized for Arabic & English'
                }
              </p>
            </div>
          </div>
        </div>

        {/* Form Section */}
        <div className="max-w-2xl mx-auto">
          <RequestDemoForm />
        </div>

        {/* What Happens Next Section */}
        <div className="max-w-3xl mx-auto mt-16 p-8 bg-gradient-to-r from-primary/5 to-primary/10 rounded-2xl border border-primary/20">
          <h2 className="text-2xl font-bold mb-6 text-center">
            {language === 'ar' ? 'ماذا سيحدث بعد ذلك؟' : 'What Happens Next?'}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-primary text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                1
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'تأكيد فوري' : 'Instant Confirmation'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar'
                  ? 'ستصلك رسالة تأكيد عبر البريد الإلكتروني'
                  : 'You\'ll receive a confirmation email'
                }
              </p>
            </div>

            <div className="text-center">
              <div className="w-12 h-12 bg-primary text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                2
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'مراجعة الفريق' : 'Team Review'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar'
                  ? 'سيراجع فريقنا طلبك خلال 24 ساعة'
                  : 'Our team reviews your request within 24 hours'
                }
              </p>
            </div>

            <div className="text-center">
              <div className="w-12 h-12 bg-primary text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                3
              </div>
              <h3 className="font-semibold mb-2">
                {language === 'ar' ? 'ابدأ التجربة' : 'Start Testing'}
              </h3>
              <p className="text-sm text-gray-600">
                {language === 'ar'
                  ? 'احصل على رابط الوصول المباشر للوحة التجريب'
                  : 'Get direct access link to demo dashboard'
                }
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
