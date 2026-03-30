import { Link } from "react-router-dom";
import logo from "@/assets/genius-ai-logo.png";
import { useLanguage } from "@/contexts/LanguageContext";

const Footer = () => {
  const { t } = useLanguage();

  return (
    <footer className="py-12 border-t border-border/50">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <img src={logo} alt="Genius AI" className="h-8 w-auto" />
          </div>
          
          <div className="flex flex-wrap items-center justify-center gap-8 text-sm text-muted-foreground">
            <a href="#features" className="hover:text-primary transition-colors duration-300">{t('nav.features')}</a>
            <a href="#about" className="hover:text-primary transition-colors duration-300">{t('nav.about')}</a>
            <a href="#contact" className="hover:text-primary transition-colors duration-300">{t('nav.contact')}</a>
            <Link to="/privacy-Policy" className="hover:text-primary transition-colors duration-300">{t('footer.privacy')}</Link>
            <Link to="/terms" className="hover:text-primary transition-colors duration-300">{t('footer.terms')}</Link>
          </div>

          <p className="text-sm text-muted-foreground">
            {t('footer.rights')}
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
