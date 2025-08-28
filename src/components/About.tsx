import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Target, Users, Lightbulb, Award } from "lucide-react";
import { useLanguage } from "@/hooks/useLanguage";
import { getTranslation } from "@/utils/translations";

const About = () => {
  const { language } = useLanguage();

  const values = [
    {
      icon: Target,
      title: getTranslation("customerCentric", language),
      description: getTranslation("customerCentricDesc", language)
    },
    {
      icon: Lightbulb,
      title: getTranslation("innovationFirst", language),
      description: getTranslation("innovationFirstDesc", language)
    },
    {
      icon: Users,
      title: getTranslation("partnership", language),
      description: getTranslation("partnershipDesc", language)
    },
    {
      icon: Award,
      title: getTranslation("excellence", language),
      description: getTranslation("excellenceDesc", language)
    }
  ];
{/*
  const team = [
    {
      name: "Alex Rodriguez",
      role: language === 'ar' ? "الرئيس التنفيذي والمؤسس المشارك" : "CEO & Co-founder",
      bio: language === 'ar' ? 
        "نائب رئيس سابق للهندسة في شركات تقنية رائدة مع أكثر من 15 عاماً في الذكاء الاصطناعي وتجربة العملاء." :
        "Former VP of Engineering at leading tech companies with 15+ years in AI and customer experience.",
      expertise: language === 'ar' ? 
        ["استراتيجية الذكاء الاصطناعي", "رؤية المنتج", "نجاح العملاء"] :
        ["AI Strategy", "Product Vision", "Customer Success"]
    },
    {
      name: "Dr. Sarah Kim",
      role: language === 'ar' ? "الرئيس التنفيذي للتكنولوجيا والمؤسس المشارك" : "CTO & Co-founder",
      bio: language === 'ar' ?
        "دكتوراه في التعلم الآلي من ستانفورد. قادت سابقاً فرق بحث الذكاء الاصطناعي في شركات تقنية كبرى." :
        "PhD in Machine Learning from Stanford. Previously led AI research teams at major tech companies.",
      expertise: language === 'ar' ?
        ["التعلم الآلي", "معالجة اللغة الطبيعية", "هندسة الأنظمة"] :
        ["Machine Learning", "NLP", "System Architecture"]
    },
    {
      name: "Michael Chen",
      role: language === 'ar' ? "نائب رئيس الهندسة" : "VP of Engineering",
      bio: language === 'ar' ?
        "خبير في الأنظمة القابلة للتوسع وبنية الذكاء الاصطناعي مع خبرة في الشركات الناشئة الناجحة." :
        "Expert in scalable systems and AI infrastructure with experience at unicorn startups.",
      expertise: language === 'ar' ?
        ["الهندسة", "DevOps", "الأمان"] :
        ["Engineering", "DevOps", "Security"]
    }
  ];*/}

  const stats = [
    {/*{ number: "500+", label: getTranslation("companiesServed", language) },
    { number: "50M+", label: getTranslation("conversationsHandled", language) },
    { number: "99.9%", label: getTranslation("uptimeGuarantee", language) },
    { number: "150+", label: getTranslation("teamMembers", language) }*/}
  ];

  return (
    <section id="about" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Company Mission */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            {getTranslation("aboutTitle", language)}
          </h2>
          <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            {getTranslation("aboutDescription", language)}
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              {/*<div className="text-4xl font-bold text-primary mb-2">{stat.number}</div>
              <div className="text-muted-foreground font-medium">{stat.label}</div>*/}
            </div>
          ))}
        </div>

        {/* Values */}
        <div className="mb-20">
          <h3 className="text-2xl font-bold text-center text-foreground mb-12">{getTranslation("ourValues", language)}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {values.map((value, index) => (
              <Card key={index} className="text-center group hover:shadow-elegant transition-all duration-300">
                <CardHeader>
                  <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-primary/20 transition-colors">
                    <value.icon className="h-8 w-8 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{value.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>{value.description}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Team */}
        {/*<div>
          <h3 className="text-2xl font-bold text-center text-foreground mb-12">{getTranslation("leadershipTeam", language)}</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <Card key={index} className="group hover:shadow-elegant transition-all duration-300">
                <CardHeader>
                  <div className="w-20 h-20 bg-gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl font-bold text-primary-foreground">
                      {member.name.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>
                  <CardTitle className="text-center">{member.name}</CardTitle>
                  <CardDescription className="text-center font-medium text-primary">
                    {member.role}
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-center space-y-4">
                  <p className="text-muted-foreground text-sm">{member.bio}</p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {member.expertise.map((skill, skillIndex) => (
                      <Badge key={skillIndex} variant="secondary" className="text-xs">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>*/}
      </div>
    </section>
  );
};

export default About;