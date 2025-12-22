import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X, Globe } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";
import logo from "@/assets/logo.png";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { language, toggleLanguage, isRTL } = useLanguage();

  const navigation = [
    { name: getTranslation("home", language), href: "./#home" },
    { name: getTranslation("caseStudies", language), href: "./#case-studies" },
    { name: getTranslation("services", language), href: "./#services" },
    { name: getTranslation("about", language), href: "./#about" },
    { name: getTranslation("contact", language), href: "./#contact" },
  ];

  return (
    <header className="fixed top-0 w-full bg-background/95 backdrop-blur-sm border-b border-border z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <img 
              src={logo} 
              alt="Genius AI" 
              className="h-12 w-auto"
            />
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex gap-10">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="text-foreground hover:text-primary transition-smooth font-medium"
              >
                {item.name}
              </a>
            ))}
          </nav>

          {/* CTA Button & Language Toggle */}
          <div className="hidden md:flex items-center space-x-4">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={toggleLanguage}
              className="gap-2"
            >
              <Globe size={16} />
              {getTranslation("language", language)}
            </Button>
            <a href="/login">
              <Button variant="outline" size="sm">
                {getTranslation("signIn", language)}
              </Button>
            </a>
            <a href="#contact">
            <Button variant="cta" size="sm">
              {getTranslation("requestDemo", language)}
            </Button></a>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-foreground hover:text-primary transition-smooth"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-card border-t border-border">
              {navigation.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className="block px-3 py-2 text-foreground hover:text-primary transition-smooth font-medium"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </a>
              ))}
              <div className="pt-4 space-y-2">
                <Button 
                  variant="ghost" 
                  onClick={toggleLanguage}
                  className="w-full gap-2"
                >
                  <Globe size={16} />
                  {getTranslation("language", language)}
                </Button>
                <a href="/login" className="block">
                  <Button variant="outline" className="w-full">
                    {getTranslation("signIn", language)}
                  </Button>
                </a>
                <Button variant="cta" className="w-full">
                  {getTranslation("requestDemo", language)}
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;