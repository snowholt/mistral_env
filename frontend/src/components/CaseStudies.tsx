import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowRight, TrendingUp, Users, Clock } from "lucide-react";
import analyticsImage from "@/assets/analytics-dashboard.jpg";
import aiServiceImage from "@/assets/ai-service-illustration.jpg";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const CaseStudies = () => {
  const { language } = useLanguage();

  const caseStudies = [
    {
      company: language === 'ar' ? "سِنا" : "S.I.N.A",
      industry: language === 'ar' ? "عميل ذكاء صناعي صوتي" : "Voice AI Agent",
      image: analyticsImage,
      challenge: language === 'ar' ? 
        "تواجه العديد من الشركات صعوبة في تقديم دعم عملاء سلس وفعّال عبر القنوات الصوتية، مما يؤدي إلى فترات انتظار طويلة وتجارب عملاء سيئة." :
        "Many businesses struggle to provide seamless, real-time customer support through voice channels, leading to long wait times and poor customer experiences.",
      solution: language === 'ar' ?
        "يسرع وكيل الذكاء الاصطناعي من الصوت إلى الصوت دعم العملاء من خلال تقديم ردود فورية ودقيقة، مما يقلل من فترات الانتظار ويعزز رضا العملاء." :
        "Our Voice-to-Voice AI Agent streamlines customer service by offering instant, accurate voice-based responses, reducing wait times and enhancing customer satisfaction.",
      results: [
        { metric: getTranslation("waitingTime", language), improvement: language === 'ar' ? "أسرع بنسبة 95%" : "95% faster", icon: Clock },
        //{ metric: getTranslation("waitingTime", language), improvement: "95%", icon: Users },
        { metric: getTranslation("costSavings", language), improvement: language === 'ar' ? "65% تكلفة أقل" : "65% Cost Reduction", icon: TrendingUp }
      ]
    },
    {
      company:  language === 'ar' ? "سِنا واتساب" : "S.I.N.A Whatsapp",
      industry: language === 'ar' ? "العميل الذكي للواتساب" : "Whatsapp Smart Agent",
      image: aiServiceImage,
      challenge: language === 'ar' ?
        "واجه الشركات صعوبة في إدارة كميات كبيرة من التفاعلات مع العملاء على منصات المراسلة مثل WhatsApp، مما يؤدي إلى الفرص الضائعة وتأخير الردود." :
        "Businesses find it difficult to manage high volumes of customer interactions on messaging platforms like WhatsApp, leading to missed opportunities and delayed responses.",
      solution: language === 'ar' ?
        "يقوم وكيل الذكاء الاصطناعي في WhatsApp بأتمتة المحادثات، مما يضمن الرد الفوري، والتفاعل الشخصي، وزيادة التفاعل مع العملاء على المنصة." :
        "The WhatsApp AI Agent automates conversations, ensuring instant replies, personalized interactions, and higher customer engagement on the platform.",
      results: [
        { metric: getTranslation("cartConversion", language), improvement: "+32%", icon: TrendingUp },
        { metric: getTranslation("supportTickets", language), improvement: language === 'ar' ? "انخفاض بنسبة 68%" : "68% reduction", icon: Users },
        { metric: getTranslation("revenueImpact", language), improvement: language === 'ar' ? "+5.2 مليون دولار" : "+$5.2M", icon: TrendingUp }
      ]
    },
    {
      company:  language === 'ar' ? "سِنا روبوت الدردشة" : "S.I.N.A Chatbot",
      industry: language === 'ar' ? "نموذج لغوي خبير" : "Subject Matter Expert LLM",
      image: aiServiceImage,
      challenge: language === 'ar' ?
        "يطالب العملاء بردود فورية على المواقع الإلكترونية والتطبيقات، لكن العديد من الشركات لا تزال تعتمد على أنظمة قديمة وبطيئة مما يزعج المستخدمين." :
        "Customers demand instant responses on websites and apps, but many companies still rely on slow, outdated systems that frustrate users.",
      solution: language === 'ar' ?
        "يوفر وكيل الذكاء الاصطناعي للدردشة ردود فعل فورية وذكية لزوار الموقع الإلكتروني، مما يعزز التفاعل ويقلل من الإحباط مع دعم آلي سريع." :
        "Our Chatbot AI Agent provides real-time, intelligent responses to website visitors, improving engagement and reducing customer frustration with fast, automated support.",
      results: [
        { metric: getTranslation("cartConversion", language), improvement: "+32%", icon: TrendingUp },
        { metric: getTranslation("supportTickets", language), improvement: language === 'ar' ? "انخفاض بنسبة 68%" : "68% reduction", icon: Users },
        { metric: getTranslation("revenueImpact", language), improvement: language === 'ar' ? "+5.2 مليون دولار" : "+$5.2M", icon: TrendingUp }
      ]
    }
  ];
{/*
  const testimonials = [
    {
      quote: language === 'ar' ?
        "وكلاء الذكاء الاصطناعي حولوا خدمة العملاء لدينا. نحن قادرون الآن على تقديم استجابات فورية ودقيقة على مدار الساعة، ونتائج رضا العملاء لم تكن أعلى من ذلك أبداً." :
        "The AI agents have transformed our customer service. We're now able to provide instant, accurate responses 24/7, and our customer satisfaction scores have never been higher.",
      author: "Sarah Johnson",
      title: language === 'ar' ? "نائب الرئيس لنجاح العملاء" : "VP of Customer Success",
      company: "TechCorp Solutions"
    },
    {
      quote: language === 'ar' ?
        "كان التنفيذ سلساً، وكان العائد على الاستثمار واضحاً خلال الشهر الأول. يمكن لفريق الدعم لدينا الآن التركيز على القضايا المعقدة بينما يتعامل الذكاء الاصطناعي مع الاستفسارات الروتينية بشكل مثالي." :
        "Implementation was seamless, and the ROI was evident within the first month. Our support team can now focus on complex issues while AI handles routine inquiries perfectly.",
      author: "Mike Chen",
      title: language === 'ar' ? "الرئيس التنفيذي للتكنولوجيا" : "CTO",
      company: "RetailMax"
    }
  ];
*/}
  return (
    <section id="case-studies" className="py-20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {getTranslation("caseStudiesTitle", language)}
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            {getTranslation("caseStudiesSubtitle", language)}
          </p>
        </div>

        {/* Case Studies */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {caseStudies.map((study, index) => (
            <div key={index} className={`"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"`}>
              <Card className={`group hover:shadow-elegant transition-all duration-300 hover:-translate-y- px-4 py-4`}>
              <div className={index % 2 === 1 ? 'lg:col-start-2' : ''}>
                <div className="space-y-4">
                  <div className="inline-block bg-primary/10 text-primary px-3 py-1 rounded-full text-sm font-medium">
                    {study.industry}
                  </div>
                  <h3 className="text-2xl font-bold text-foreground">{study.company}</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">{getTranslation("challenge", language)}</h4>
                      <p className="text-muted-foreground">{study.challenge}</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">{getTranslation("solution", language)}</h4>
                      <p className="text-muted-foreground">{study.solution}</p>
                    </div>
                  </div>
                  {/*<div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
                    {study.results.map((result, resultIndex) => (
                      <div key={resultIndex} className="text-center p-4 bg-card rounded-lg border">
                        <result.icon className="h-8 w-8 text-primary mx-auto mb-2" />
                        <div className="text-2xl font-bold text-primary mb-1">{result.improvement}</div>
                        <div className="text-sm text-muted-foreground">{result.metric}</div>
                      </div>
                    ))}
                  </div>*/}
                </div>
              </div>
              {/*<div className={index % 2 === 1 ? 'lg:col-start-1' : ''}>
                <img
                  src={study.image}
                  alt={`${study.company} case study`}
                  className="w-full h-auto rounded-lg shadow-elegant"
                />
              </div>*/}</Card>
            </div>
          ))}
        </div>

        {/* Testimonials */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/*{testimonials.map((testimonial, index) => (
            <Card key={index} className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <blockquote className="text-lg text-foreground mb-4">
                  "{testimonial.quote}"
                </blockquote>
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center mr-4">
                    <span className="text-primary font-bold">
                      {testimonial.author.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>
                  <div>
                    <div className="font-semibold text-foreground">{testimonial.author}</div>
                    <div className="text-sm text-muted-foreground">{testimonial.title}, {testimonial.company}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}*/}
        </div>

        {/*<div className="text-center mt-12">
          <Button variant="cta" size="lg" className="group">
            {getTranslation("viewMoreCaseStudies", language)}
            <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
          </Button>
        </div>*/}
      </div>
    </section>
  );
};

export default CaseStudies;