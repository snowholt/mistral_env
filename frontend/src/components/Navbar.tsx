import { useState } from "react";
import { Link } from "react-router-dom";
import { Moon, Sun, Globe, Menu, X, LogIn } from "lucide-react";
import logo from "@/assets/genius-ai-logo.png";
import { useTheme } from "@/contexts/ThemeContext";
import { useLanguage } from "@/contexts/LanguageContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: "#features", label: t('nav.features') },
    { href: "#products", label: t('nav.products') },
    { href: "#about", label: t('nav.about') },
    { href: "#contact", label: t('nav.contact') },
  ];

  const closeMobileMenu = () => setIsMobileMenuOpen(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border/50">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src={logo} alt="Genius AI" className="h-10 w-auto" />
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-muted-foreground hover:text-primary transition-colors duration-300"
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-3">
          {/* Language Selector */}
          <DropdownMenu>
            <DropdownMenuTrigger className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border/50 hover:border-primary/50 transition-colors">
              <Globe className="w-4 h-4" />
              <span className="text-sm">{language === 'en' ? 'EN' : 'عربي'}</span>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setLanguage('en')}>
                English
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setLanguage('ar')}>
                العربية
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg border border-border/50 hover:border-primary/50 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </button>

          {/* Sign In Button */}
          <Link
            to="/login"
            className="hidden md:inline-flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium border border-border/50 hover:border-primary/50 hover:text-primary transition-colors duration-300"
          >
            <LogIn className="w-4 h-4" />
            {language === 'en' ? 'Sign In' : 'تسجيل الدخول'}
          </Link>

          {/* Desktop CTA */}
          <a
            href="#contact"
            className="hidden md:inline-flex bg-gradient-primary text-primary-foreground px-5 py-2.5 rounded-lg font-medium hover:opacity-90 transition-opacity duration-300 glow-primary"
          >
            {t('nav.getStarted')}
          </a>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg border border-border/50 hover:border-primary/50 transition-colors"
            aria-label="Toggle mobile menu"
          >
            {isMobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-background/95 backdrop-blur-lg border-t border-border/50">
          <div className="container mx-auto px-6 py-4 flex flex-col gap-4">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={closeMobileMenu}
                className="text-foreground hover:text-primary transition-colors duration-300 py-2"
              >
                {link.label}
              </a>
            ))}
            <Link
              to="/login"
              onClick={closeMobileMenu}
              className="flex items-center justify-center gap-2 border border-border/50 text-foreground px-5 py-3 rounded-lg font-medium hover:border-primary/50 hover:text-primary transition-colors duration-300"
            >
              <LogIn className="w-4 h-4" />
              {language === 'en' ? 'Sign In' : 'تسجيل الدخول'}
            </Link>
            <a
              href="#contact"
              onClick={closeMobileMenu}
              className="bg-gradient-primary text-primary-foreground px-5 py-3 rounded-lg font-medium hover:opacity-90 transition-opacity duration-300 glow-primary text-center"
            >
              {t('nav.getStarted')}
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
