import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Menu, X, Globe, Sun, Moon } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";
import logo from "@/assets/logo.png";
import { Link } from "react-router-dom";
import { useTheme } from "@/contexts/ThemeContext";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { language, toggleLanguage, isRTL } = useLanguage();
  const { theme, toggleTheme } = useTheme();

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

          {/* CTA Button, Theme Toggle & Language Toggle */}
          <div className="hidden md:flex items-center space-x-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleTheme}
              className="gap-2"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              {theme === "dark" ? "Light" : "Dark"}
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={toggleLanguage}
              className="gap-2"
            >
              <Globe size={16} />
              {getTranslation("language", language)}
            </Button>
            <Link to="/login">
              <Button variant="outline" size="sm">
                {getTranslation("signIn", language)}
              </Button>
            </Link>
            <Link to="/request-demo">
              <Button variant="cta" size="sm">
                {getTranslation("requestDemo", language)}
              </Button>
            </Link>
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
                  onClick={toggleTheme}
                  className="w-full gap-2"
                  aria-label="Toggle theme"
                >
                  {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
                  {theme === "dark" ? "Light" : "Dark"}
                </Button>
                <Button 
                  variant="ghost" 
                  onClick={toggleLanguage}
                  className="w-full gap-2"
                >
                  <Globe size={16} />
                  {getTranslation("language", language)}
                </Button>
                <Link to="/login" className="block">
                  <Button variant="outline" className="w-full">
                    {getTranslation("signIn", language)}
                  </Button>
                </Link>
                <Link to="/request-demo" className="block">
                  <Button variant="cta" className="w-full">
                    {getTranslation("requestDemo", language)}
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;