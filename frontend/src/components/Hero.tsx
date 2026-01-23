import logo from "@/assets/genius-ai-logo.png";
import { useLanguage } from "@/contexts/LanguageContext";

const Hero = () => {
  const { t } = useLanguage();

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-32">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse-glow" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/15 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: '1.5s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-secondary/30 rounded-full blur-3xl" />
      </div>

      {/* Grid pattern overlay */}
      <div 
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(hsl(var(--primary)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--primary)) 1px, transparent 1px)`,
          backgroundSize: '60px 60px'
        }}
      />

      <div className="relative container mx-auto px-6 text-center">
        <div className="flex flex-col items-center gap-8 animate-fade-in">
          {/* Logo with padding to prevent animation clipping */}
          <div className="py-6">
            <img 
              src={logo} 
              alt="Genius AI" 
              className="h-32 md:h-40 w-auto animate-float drop-shadow-2xl"
            />
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-display font-bold max-w-4xl leading-tight">
            {t('hero.title')}{' '}
            <span className="text-gradient glow-text">{t('hero.titleHighlight')}</span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl leading-relaxed">
            {t('hero.subtitle')}
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <a 
              href="#contact" 
              className="bg-gradient-primary text-primary-foreground px-8 py-4 rounded-xl font-semibold text-lg hover:opacity-90 transition-all duration-300 glow-primary hover:scale-105"
            >
              {t('hero.cta')}
            </a>
            <a 
              href="#products" 
              className="border border-primary/50 text-foreground px-8 py-4 rounded-xl font-semibold text-lg hover:bg-primary/10 transition-all duration-300"
            >
              {t('hero.explore')}
            </a>
          </div>

          {/* Trust badges */}
          <div className="mt-12 flex flex-wrap items-center justify-center gap-8 text-muted-foreground text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span>{t('hero.enterprise')}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" style={{ animationDelay: '0.5s' }} />
              <span>{t('hero.uptime')}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" style={{ animationDelay: '1s' }} />
              <span>{t('hero.support')}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
