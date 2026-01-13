import { Separator } from "@/components/ui/separator";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";
import { Link } from "react-router-dom";

const Footer = () => {
  const { language } = useLanguage();

  const getFooterLinks = () => ({
    [getTranslation("company", language)]: [
      { name: getTranslation("home", language), href: "/#home" },
      { name: getTranslation("caseStudies", language), href: "/#case-studies" },
      { name: getTranslation("services", language), href: "/#services" },
      { name: getTranslation("about", language), href: "/#about" },
      { name: getTranslation("contact", language), href: "/#contact" }
    ],
    [getTranslation("legal", language)]: [
      { name: getTranslation("privacyPolicy", language), href: "/privacy-policy" },
      { name: getTranslation("termsOfService", language), href: "/terms" },
    ]
  });

  const footerLinks = getFooterLinks();

  return (
    <footer className="bg-secondary text-secondary-foreground">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Footer Content */}
        <div className="py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8">
            {/* Company Info */}
            <div className="lg:col-span-2">
              <div className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
                Genius AI
              </div>
              <p className="text-muted-foreground mb-6 max-w-md">
                {getTranslation("companyInfo", language)}
              </p>
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
