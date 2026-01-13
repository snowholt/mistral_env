import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquare, Brain, BarChart3, Clock, Shield, Zap } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const Terms = () => {
  const { language } = useLanguage();
  const terms = [
    {
      title: getTranslation("term_1_t", language),
      content: [
        getTranslation("term_1_c", language),
      ]
    },
    {
      title: getTranslation("term_2_t", language),
      content: [
        getTranslation("term_2_c", language),
      ]
    },
    {
      title: getTranslation("term_3_t", language),
      content: [
        getTranslation("term_3_c", language),
      ]
    },
    {
      title: getTranslation("term_4_t", language),
      content: [
        getTranslation("term_4_c", language),
      ]
    },
    {
      title: getTranslation("term_5_t", language),
      content: [
        getTranslation("term_5_c", language),
      ]
    },
    {
      title: getTranslation("term_6_t", language),
      content: [
        getTranslation("term_6_c", language),
      ]
    },
    {
      title: getTranslation("term_7_t", language),
      content: [
        getTranslation("term_7_c", language),
      ]
    },
    {
      title: getTranslation("term_8_t", language),
      content: [
        getTranslation("term_8_c", language),
      ]
    },
    {
      title: getTranslation("term_9_t", language),
      content: [
        getTranslation("term_9_c", language),
      ]
    },
    {
      title: getTranslation("term_10_t", language),
      content: [
        getTranslation("term_10_c", language),
      ]
    }
  ];

  return (
    <section id="terms" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <h6 className="max-w-3xl mx-auto mt-1" style={{ color: "rgba(132, 132, 132, 0.47)" }}>15/12/2025
        </h6>
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {getTranslation("termTitle", language)}
          </h2>
          {terms.map((term, termIndex) => (
            // Use termIndex as the key for the outer element
            <div key={termIndex}>
              
              {/* 1. Render the Title */}
              <h3 className="text-xl font-bold max-w-3xl mx-auto">
                {term.title}
              </h3>
              
              {/* 2. Map over the 'content' array for individual lines */}
              {term.content.map((contentLine, contentIndex) => (
                // Use the contentIndex (or a combination) as the key for the inner element
                <p 
                  key={`${termIndex}-${contentIndex}`}
                  className="text-lg text-muted-foreground max-w-3xl mx-auto mt-1" // Added margin-top for separation
                >
                  {contentLine}
                </p>
              ))}
              
            </div>
          ))}
        </div>

 
      </div>
    </section>
  );
};

export default Terms;