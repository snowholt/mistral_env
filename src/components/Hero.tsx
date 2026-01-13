import { Button } from "@/components/ui/button";
import { ArrowRight, Play } from "lucide-react";
import heroImage from "@/assets/hero-ai-customer-service.jpg";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";
import { Link } from "react-router-dom";

const Hero = () => {
  const { language } = useLanguage();

  return (
    <section id="home" className="relative min-h-screen flex items-center pt-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Content */}
          <div className="text-center lg:text-left">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground leading-tight">
              {getTranslation("heroTitle", language)}
            </h1>
            <p className="mt-6 text-xl text-muted-foreground max-w-2xl">
              {getTranslation("heroDescription", language)}
            </p>
            <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link to="/request-demo">
                <Button variant="hero" size="lg" className="group w-full sm:w-auto">
                  {getTranslation("requestDemoBtn", language)}
                  <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link to="#contact">
                <Button variant="outline" size="lg" className="group w-full sm:w-auto">
                  {getTranslation("contactUs", language)}
                </Button>
              </Link>
            </div>
            <div className="mt-12 grid grid-cols-3 gap-8 text-center lg:text-left">
              {<div>
                <div className="text-3xl font-bold text-primary">95%</div>
                <div className="text-sm text-muted-foreground">{getTranslation("waitingTime", language)}</div>
              </div>}
              <div>
                <div className="text-3xl font-bold text-primary">24/7</div>
                <div className="text-sm text-muted-foreground">{getTranslation("availability", language)}</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-primary">65%</div>
                <div className="text-sm text-muted-foreground">{getTranslation("costReduction", language)}</div>
              </div>
            </div>
          </div>

          {/* Hero Image */}
          <div className="relative">
            <div className="relative z-10">
              <img
                src={heroImage}
                alt="AI Customer Service Interface"
                className="w-full h-auto rounded-lg shadow-elegant"
              />
            </div>
            {/* Background gradient blur */}
            <div className="absolute inset-0 bg-gradient-primary opacity-20 blur-3xl transform scale-110"></div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;