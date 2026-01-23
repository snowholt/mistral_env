import { useLanguage } from "@/contexts/LanguageContext";
import { BadgeCheck, Target, Earth } from "lucide-react";

const About = () => {
  const { t } = useLanguage();

  const stats = [
    { value: 'about.title1', labelKey: 'about.stat1' },
    { value: 'about.title2', labelKey: 'about.stat2' },
    { value: 'about.title3', labelKey: 'about.stat3' },
    { value: 'about.title4', labelKey: 'about.stat4' }
  ];

  return (
    <section id="about" className="py-24 relative">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-display font-bold mb-4">
            {t('about.title')} <span className="text-gradient">{t('about.titleHighlight')}</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            {t('about.subtitle')}
          </p>
        </div>

        {/*<div className="grid lg:grid-cols-2 gap-16 items-start">*/}
        <div className="grid lg:grid-cols-1 gap-16 items-start">
          {/* Content */}
          <div>
            <h3 className="text-2xl font-display font-semibold mb-4">{t('about.mission')}</h3>
            <p className="text-muted-foreground text-lg leading-relaxed mb-8">
              {t('about.missionText')}
            </p>

            <div className="grid sm:grid-cols-3 gap-4">
              <div className="bg-card/50 backdrop-blur-sm border border-border/50 rounded-xl p-4">
                <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-300 glow-primary">
                  <BadgeCheck className="w-7 h-7 text-primary-foreground" />
                </div>
                <h4 className="font-semibold mb-1">{t('about.certified')}</h4>
                <p className="text-sm text-muted-foreground">{t('about.certifiedText')}</p>
              </div>
              <div className="bg-card/50 backdrop-blur-sm border border-border/50 rounded-xl p-4">
                <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-300 glow-primary">
                  <Target className="w-7 h-7 text-primary-foreground" />
                </div>
                <h4 className="font-semibold mb-1">{t('about.vision')}</h4>
                <p className="text-sm text-muted-foreground">{t('about.visionText')}</p>
              </div>
              <div className="bg-card/50 backdrop-blur-sm border border-border/50 rounded-xl p-4">
                <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center mb-2 group-hover:scale-110 transition-transform duration-300 glow-primary">
                  <Earth className="w-7 h-7 text-primary-foreground" />
                </div>
                <h4 className="font-semibold mb-1">{t('about.expertise')}</h4>
                <p className="text-sm text-muted-foreground">{t('about.expertiseText')}</p>
              </div>
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-6">
            {stats.map((stat, index) => (
              <div
                key={index}
                className="bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 text-center hover:border-primary/30 transition-colors duration-300"
              >
                <div className="text-4xl md:text-4xl font-display font-bold text-gradient mb-2">
                  {t(stat.value)}
                </div>
                <div className="text-muted-foreground">{t(stat.labelKey)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
