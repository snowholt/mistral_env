import { Brain, MessageSquare, Eye, TrendingUp, Cog, Shield, BarChart, Clock, Power, Zap } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

const Features = () => {
  const { t } = useLanguage();

  const features = [
    {
      icon: Brain,
      titleKey: 'features.ml.title',
      descKey: 'features.ml.desc'
    },
    {
      icon: MessageSquare,
      titleKey: 'features.nlp.title',
      descKey: 'features.nlp.desc'
    },
    {
      icon: BarChart,
      titleKey: 'features.vision.title',
      descKey: 'features.vision.desc'
    },
    {
      icon: Clock,
      titleKey: 'features.analytics.title',
      descKey: 'features.analytics.desc'
    },
    {
      icon: Zap,
      titleKey: 'features.automation.title',
      descKey: 'features.automation.desc'
    },
    {
      icon: Shield,
      titleKey: 'features.security.title',
      descKey: 'features.security.desc'
    }
  ];

  return (
    <section id="features" className="py-24 relative">
      {/* Background accent */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-display font-bold mb-4">
            {t('features.title')} <span className="text-gradient">{t('features.titleHighlight')}</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            {t('features.subtitle')}
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-500 hover:bg-card/80"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="w-14 h-14 bg-gradient-primary rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 glow-primary">
                <feature.icon className="w-7 h-7 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-display font-semibold mb-3">{t(feature.titleKey)}</h3>
              <p className="text-muted-foreground leading-relaxed">{t(feature.descKey)}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
