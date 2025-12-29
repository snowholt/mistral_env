/**
 * Dashboard Home Page
 * 
 * Overview with stats, recent activity, and quick actions.
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  MessageSquare,
  Users,
  Bot,
  TrendingUp,
  ArrowRight,
  Clock,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { useAuth } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import api from '@/lib/api';

const translations = {
  en: {
    welcome: 'Welcome back',
    overview: 'Your business overview',
    totalMessages: 'Total Messages',
    activeChats: 'Active Chats',
    responseRate: 'Response Rate',
    avgResponseTime: 'Avg Response Time',
    quickActions: 'Quick Actions',
    viewInbox: 'View Inbox',
    configureAgent: 'Configure AI Agent',
    addKnowledge: 'Add Knowledge',
    viewMetrics: 'View Metrics',
    recentActivity: 'Recent Activity',
    noActivity: 'No recent activity',
    getStarted: 'Get Started',
    setupSteps: 'Complete these steps to get your AI assistant running',
    step1Title: 'Connect WhatsApp',
    step1Desc: 'Link your WhatsApp Business account',
    step2Title: 'Configure AI Agent',
    step2Desc: 'Set up your AI assistant personality',
    step3Title: 'Add Knowledge',
    step3Desc: 'Upload documents to train your AI',
    step4Title: 'Go Live',
    step4Desc: 'Start receiving messages',
    completed: 'Completed',
    pending: 'Pending',
    today: 'Today',
    yesterday: 'Yesterday',
    thisWeek: 'This Week',
  },
  ar: {
    welcome: 'مرحباً بعودتك',
    overview: 'نظرة عامة على عملك',
    totalMessages: 'إجمالي الرسائل',
    activeChats: 'المحادثات النشطة',
    responseRate: 'معدل الاستجابة',
    avgResponseTime: 'متوسط وقت الاستجابة',
    quickActions: 'إجراءات سريعة',
    viewInbox: 'عرض صندوق الوارد',
    configureAgent: 'إعداد الوكيل الذكي',
    addKnowledge: 'إضافة معرفة',
    viewMetrics: 'عرض المقاييس',
    recentActivity: 'النشاط الأخير',
    noActivity: 'لا يوجد نشاط حديث',
    getStarted: 'ابدأ الآن',
    setupSteps: 'أكمل هذه الخطوات لتشغيل مساعدك الذكي',
    step1Title: 'ربط واتساب',
    step1Desc: 'اربط حساب واتساب للأعمال الخاص بك',
    step2Title: 'إعداد الوكيل الذكي',
    step2Desc: 'حدد شخصية مساعدك الذكي',
    step3Title: 'إضافة المعرفة',
    step3Desc: 'ارفع المستندات لتدريب الذكاء الاصطناعي',
    step4Title: 'البدء',
    step4Desc: 'ابدأ في استقبال الرسائل',
    completed: 'مكتمل',
    pending: 'قيد الانتظار',
    today: 'اليوم',
    yesterday: 'أمس',
    thisWeek: 'هذا الأسبوع',
  },
};

interface DashboardStats {
  total_messages: number;
  active_chats: number;
  response_rate: number;
  avg_response_time: string;
}

export default function DashboardHome() {
  const { user } = useAuth();
  const { language, isRTL } = useLanguage();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const t = translations[language as keyof typeof translations] || translations.en;

  useEffect(() => {
    // Fetch real stats from backend API
    const fetchStats = async () => {
      try {
        const data = await api.get<DashboardStats>('/api/v1/dashboard/stats');
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        // Fallback to zeros if API fails
        setStats({
          total_messages: 0,
          active_chats: 0,
          response_rate: 0,
          avg_response_time: '-',
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
  }, []);

  const statCards = [
    { title: t.totalMessages, value: stats?.total_messages || 0, icon: MessageSquare, color: 'text-blue-500' },
    { title: t.activeChats, value: stats?.active_chats || 0, icon: Users, color: 'text-green-500' },
    { title: t.responseRate, value: `${stats?.response_rate || 0}%`, icon: TrendingUp, color: 'text-purple-500' },
    { title: t.avgResponseTime, value: stats?.avg_response_time || '-', icon: Clock, color: 'text-orange-500' },
  ];

  const setupSteps = [
    { title: t.step1Title, desc: t.step1Desc, completed: false, href: '/app/businesses' },
    { title: t.step2Title, desc: t.step2Desc, completed: false, href: '/app/agent' },
    { title: t.step3Title, desc: t.step3Desc, completed: false, href: '/app/knowledge-base' },
    { title: t.step4Title, desc: t.step4Desc, completed: false, href: '/app/settings' },
  ];

  return (
    <div className={`space-y-6 ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Welcome Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          {t.welcome}, {user?.full_name?.split(' ')[0] || 'there'}! 👋
        </h1>
        <p className="text-muted-foreground">{t.overview}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-8 w-20" />
              ) : (
                <div className="text-2xl font-bold">{stat.value}</div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Two Column Layout */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>{t.quickActions}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            <Link to="/app/inbox">
              <Button variant="outline" className="w-full justify-between">
                {t.viewInbox}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/app/agent">
              <Button variant="outline" className="w-full justify-between">
                {t.configureAgent}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/app/knowledge-base">
              <Button variant="outline" className="w-full justify-between">
                {t.addKnowledge}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        {/* Getting Started */}
        <Card>
          <CardHeader>
            <CardTitle>{t.getStarted}</CardTitle>
            <CardDescription>{t.setupSteps}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {setupSteps.map((step, index) => (
              <Link key={index} to={step.href}>
                <div className="flex items-center gap-3 rounded-lg border p-3 transition-colors hover:bg-muted">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full ${
                    step.completed ? 'bg-green-100' : 'bg-muted'
                  }`}>
                    {step.completed ? (
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                    ) : (
                      <span className="text-sm font-medium text-muted-foreground">{index + 1}</span>
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium">{step.title}</p>
                    <p className="text-xs text-muted-foreground">{step.desc}</p>
                  </div>
                  <Badge variant={step.completed ? 'default' : 'secondary'}>
                    {step.completed ? t.completed : t.pending}
                  </Badge>
                </div>
              </Link>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
