import { Phone, MessageCircle, Bot, Mic } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

const Products = () => {
  const { t } = useLanguage();

  const products = [
    {
      icon: Mic,
      titleKey: 'products.voice.title',
      descKey: 'products.voice.desc',
      tag: t('products.voice.tag'),
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: MessageCircle,
      titleKey: 'products.whatsapp.title',
      descKey: 'products.whatsapp.desc',
      tag: t('products.whatsapp.tag'),
      gradient: 'from-green-500 to-emerald-500',
    },
    /*{
      icon: Bot,
      titleKey: 'products.chatbot.title',
      descKey: 'products.chatbot.desc',
      tag: t('products.chatbot.tag'),
      gradient: 'from-purple-500 to-pink-500',
    },*/
  ];

  return (
    <section id="products" className="py-24 relative">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-display font-bold mb-4">
            {t('products.title')} <span className="text-gradient">{t('products.titleHighlight')}</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            {t('products.subtitle')}
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {products.map((product, index) => (
            <div 
              key={index}
              className="group bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 hover:border-primary/50 transition-all duration-500 hover:bg-card/80 relative overflow-hidden"
            >
              {/* Tag */}
              <div className="absolute top-4 end-4">
                <span className={`text-xs font-medium px-3 py-1 rounded-full bg-gradient-to-r ${product.gradient} text-white`}>
                  {product.tag}
                </span>
              </div>

              {/* Icon */}
              <div className={`w-16 h-16 bg-gradient-to-br ${product.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <product.icon className="w-8 h-8 text-white" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-display font-bold mb-3">
                {t(product.titleKey)}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {t(product.descKey)}
              </p>

              {/* Hover gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            </div>
          ))}
        </div>

        {/* Features Grid */}
        <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
          {[
            { value: '95%', labelKey: 'products.stat1' },
            { value: '24/7', labelKey: 'products.stat2' },
            { value: '65%', labelKey: 'products.stat3' },
            { value: '<1s', labelKey: 'products.stat4' },
          ].map((stat, index) => (
            <div key={index} className="text-center p-4">
              <div className="text-2xl md:text-3xl font-display font-bold text-primary mb-1">
                {stat.value}
              </div>
              <div className="text-sm text-muted-foreground">
                {t(stat.labelKey)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Products;