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
  Sparkles,
} from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
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
    // Guest status translations
    demoAccount: 'Demo Account',
    active: 'Active',
    expired: 'Expired',
    limited: 'Limited',
    conversations: 'Conversations',
    timeRemaining: 'Time Remaining',
    days: 'days',
    conversationsRemaining: 'conversations remaining',
    noConversationsRemaining: 'No conversations remaining',
    expiringSoon: '⚠️ Expiring soon!',
    expiresOn: 'Expires on',
    demoExpiredTitle: 'Demo access has expired',
    trialEnded: 'Your free trial period has ended',
    limitReached: 'You have reached the maximum conversations limit',
    upgradeMessage: 'Enjoy unlimited features with the full plan',
    upgradeNow: 'Upgrade Now',
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
    // Guest status translations
    demoAccount: 'حساب تجريبي',
    active: 'نشط',
    expired: 'منتهي',
    limited: 'محدود',
    conversations: 'المحادثات',
    timeRemaining: 'الوقت المتبقي',
    days: 'يوم',
    conversationsRemaining: 'محادثة متبقية',
    noConversationsRemaining: 'لا توجد محادثات متبقية',
    expiringSoon: '⚠️ ينتهي قريباً!',
    expiresOn: 'تنتهي في',
    demoExpiredTitle: 'انتهت صلاحية الوصول التجريبي',
    trialEnded: 'انتهت فترة التجربة المجانية',
    limitReached: 'وصلت إلى الحد الأقصى للمحادثات',
    upgradeMessage: 'استمتع بمزايا غير محدودة مع الخطة الكاملة',
    upgradeNow: 'ترقية الحساب',
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

  // Guest user status (only for users with role=guest)
  const isGuestUser = user?.role === 'guest';
  const conversationProgress = isGuestUser && user.max_conversations 
    ? ((user.conversations_used || 0) / user.max_conversations) * 100 
    : 0;
  const isExpiringSoon = isGuestUser && (user.days_remaining ?? 0) <= 2;
  const isExpired = isGuestUser && user.is_expired;
  const isLimitReached = isGuestUser && user.is_limit_reached;

  // Determine alert variant for guest banner
  const getGuestAlertVariant = (): "default" | "destructive" => {
    if (isExpired || isLimitReached) return "destructive";
    return "default";
  };

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

      {/* Guest User Status Banner */}
      {isGuestUser && user && (
        <Alert variant={getGuestAlertVariant()}>
          <Sparkles className="h-4 w-4" />
          <AlertTitle className="flex items-center justify-between">
            <span className="font-semibold">{t.demoAccount}</span>
            <Badge variant={user.can_access_demo ? "default" : "secondary"}>
              {user.can_access_demo
                ? t.active
                : isExpired
                  ? t.expired
                  : t.limited
              }
            </Badge>
          </AlertTitle>
          <AlertDescription>
            <div className="mt-3 space-y-4">
              {/* Usage Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Conversations Used */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4" />
                      <span className="font-medium">{t.conversations}</span>
                    </div>
                    <span className="font-bold">
                      {user.conversations_used || 0} / {user.max_conversations || 0}
                    </span>
                  </div>
                  <Progress value={conversationProgress} className="h-2" />
                  <p className="text-xs text-muted-foreground">
                    {(user.conversations_remaining ?? 0) > 0
                      ? `${user.conversations_remaining} ${t.conversationsRemaining}`
                      : t.noConversationsRemaining
                    }
                  </p>
                </div>

                {/* Time Remaining */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      <span className="font-medium">{t.timeRemaining}</span>
                    </div>
                    <span className="font-bold">
                      {user.days_remaining || 0} {t.days}
                    </span>
                  </div>
                  <Progress 
                    value={Math.min(100, ((user.days_remaining || 0) / 7) * 100)} 
                    className="h-2" 
                  />
                  <p className="text-xs text-muted-foreground">
                    {isExpiringSoon && !isExpired
                      ? t.expiringSoon
                      : isExpired
                        ? t.expired
                        : user.expires_at 
                          ? `${t.expiresOn} ${new Date(user.expires_at).toLocaleDateString(language === 'ar' ? 'ar' : 'en')}`
                          : ''
                    }
                  </p>
                </div>
              </div>

              {/* Warning Messages */}
              {(isExpired || isLimitReached) && (
                <div className="flex items-start gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
                  <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-destructive">
                      {t.demoExpiredTitle}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {isExpired ? t.trialEnded : t.limitReached}
                    </p>
                  </div>
                </div>
              )}

              {/* Upgrade CTA */}
              <div className="flex items-center justify-between pt-2 border-t">
                <p className="text-sm">{t.upgradeMessage}</p>
                <Link to="/app/billing">
                  <Button size="sm" variant="default" className="gap-2">
                    <TrendingUp className="h-4 w-4" />
                    {t.upgradeNow}
                  </Button>
                </Link>
              </div>
            </div>
          </AlertDescription>
        </Alert>
      )}

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
