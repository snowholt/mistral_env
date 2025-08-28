import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Mail, Linkedin, Twitter, Github } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const Footer = () => {
  const { language } = useLanguage();

  const getFooterLinks = () => ({
    /*[getTranslation("product", language)]: [
      { name: getTranslation("features", language), href: "#services" },
      { name: getTranslation("integrations", language), href: "#" },
      { name: language === 'ar' ? "وثائق API" : "API Documentation", href: "#" },
      { name: getTranslation("pricing", language), href: "#" }
    ],*/
    [getTranslation("company", language)]: [
      { name: getTranslation("about", language), href: "#about" },
      //{ name: getTranslation("careers", language), href: "#" },
      //{ name: getTranslation("press", language), href: "#" },
      { name: getTranslation("contact", language), href: "#contact" },
      { name: getTranslation("solution", language), href: "#case-studies" }
    ],
    /*[getTranslation("resources", language)]: [
      { name: getTranslation("blog", language), href: "#" },
      { name: getTranslation("helpCenter", language), href: "#" },
      { name: getTranslation("community", language), href: "#" }
    ],*/
    /*[getTranslation("legal", language)]: [
      { name: getTranslation("privacyPolicy", language), href: "#" },
      { name: getTranslation("termsOfService", language), href: "#" },
      { name: getTranslation("cookiePolicy", language), href: "#" },
      { name: getTranslation("security", language), href: "#" }
    ]*/
  });

  const footerLinks = getFooterLinks();

  /*const socialLinks = [
    { icon: Twitter, href: "#", label: "Twitter" },
    { icon: Linkedin, href: "#", label: "LinkedIn" },
    { icon: Github, href: "#", label: "GitHub" },
    { icon: Mail, href: "#", label: "Email" }
  ];*/

  return (
    <footer className="bg-secondary text-secondary-foreground">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Newsletter Section */}
        {/*<div className="py-12 border-b border-border/20">
          <div className="max-w-2xl mx-auto text-center">
            <h3 className="text-2xl font-bold mb-4">
              {getTranslation("newsletterTitle", language)}
            </h3>
            <p className="text-muted-foreground mb-6">
              {getTranslation("newsletterDesc", language)}
            </p>
            <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
              <Input 
                type="email" 
                placeholder={getTranslation("enterEmail", language)}
                className="flex-1 bg-background"
              />
              <Button variant="cta">
                {getTranslation("subscribe", language)}
              </Button>
            </div>
          </div>
        </div>*/}

        {/* Main Footer Content */}
        <div className="py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8">
            {/* Company Info */}
            <div className="lg:col-span-2">
              <div className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
                AI Agent Pro
              </div>
              <p className="text-muted-foreground mb-6 max-w-md">
                {getTranslation("companyInfo", language)}
              </p>
              <div className="flex space-x-4">
                {/*{socialLinks.map((social, index) => (
                  <a
                    key={index}
                    href={social.href}
                    aria-label={social.label}
                    className="w-10 h-10 bg-background/10 rounded-lg flex items-center justify-center hover:bg-primary hover:text-primary-foreground transition-colors"
                  >
                    <social.icon className="h-5 w-5" />
                  </a>
                ))}*/}
              </div>
            </div>

            {/* Footer Links */}
            {Object.entries(footerLinks).map(([category, links]) => (
              <div key={category}>
                <h4 className="font-semibold mb-4">{category}</h4>
                <ul className="space-y-3">
                  {links.map((link, index) => (
                    <li key={index}>
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-primary transition-colors text-sm"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <Separator className="bg-border/20" />

        {/* Bottom Footer */}
        <div className="py-6 flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
          <div className="mb-4 md:mb-0">
            {getTranslation("copyright", language)}
          </div>
          <div className="flex items-center space-x-6">
            <span>{getTranslation("builtWith", language)}</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;