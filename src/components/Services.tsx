import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquare, Brain, BarChart3, Clock, Shield, Zap } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const Services = () => {
  const { language } = useLanguage();

  const services = [
    {
      icon: MessageSquare,
      title: getTranslation("intelligentChatTitle", language),
      description: getTranslation("intelligentChatDesc", language),
      features: [
        getTranslation("naturalLanguage", language),
        getTranslation("multiLanguage", language),
        getTranslation("contextualConversations", language)
      ]
    },
    {
      icon: Brain,
      title: getTranslation("smartAutomationTitle", language),
      description: getTranslation("smartAutomationDesc", language),
      features: [
        getTranslation("workflowAutomation", language),
        getTranslation("smartRouting", language),
        getTranslation("humanHandoff", language)
      ]
    },
    {
      icon: BarChart3,
      title: getTranslation("analyticsTitle", language),
      description: getTranslation("analyticsDesc", language),
      features: [
        getTranslation("realTimeMetrics", language),
        getTranslation("performanceInsights", language),
        getTranslation("customReporting", language)
      ]
    },
    {
      icon: Clock,
      title: getTranslation("availabilityTitle", language),
      description: getTranslation("availabilityDesc", language),
      features: [
        getTranslation("alwaysOnline", language),
        getTranslation("globalTimezone", language),
        getTranslation("instantResponses", language)
      ]
    },
    {
      icon: Shield,
      title: getTranslation("securityTitle", language),
      description: getTranslation("securityDesc", language),
      features: [
        getTranslation("dataEncryption", language),
        getTranslation("gdprCompliant", language),
        getTranslation("socCertified", language)
      ]
    },
    {
      icon: Zap,
      title: getTranslation("integrationTitle", language),
      description: getTranslation("integrationDesc", language),
      features: [
        getTranslation("apiIntegration", language),
        getTranslation("pluginSupport", language),
        getTranslation("easyDeployment", language)
      ]
    }
  ];

  return (
    <section id="services" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {getTranslation("servicesTitle", language)}
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            {getTranslation("servicesSubtitle", language)}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <Card key={index} className="group hover:shadow-elegant transition-all duration-300 hover:-translate-y-1">
              <CardHeader>
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <service.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-xl font-semibold">{service.title}</CardTitle>
                <CardDescription className="text-muted-foreground">
                  {service.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {service.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-center text-sm text-muted-foreground">
                      <div className="w-1.5 h-1.5 bg-primary rounded-full mr-3"></div>
                      {feature}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Services;