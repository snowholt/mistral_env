/**
 * WhatsApp Settings Page
 * 
 * Configure WhatsApp Business account settings and AI agent behavior.
 */

import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  MessageSquare,
  Bot,
  Settings,
  Bell,
  Clock,
  Loader2,
  Save,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Volume2,
} from 'lucide-react';
import { api } from '@/lib/api';
import { useLanguage } from '@/hooks/useLanguage';
import { useToast } from '@/components/ui/use-toast';

const translations = {
  en: {
    title: 'WhatsApp Settings',
    description: 'Configure your WhatsApp Business account and AI agent',
    overview: 'Overview',
    aiSettings: 'AI Agent',
    notifications: 'Notifications',
    businessHours: 'Business Hours',
    // Overview
    accountInfo: 'Account Information',
    phoneNumber: 'Phone Number',
    verifiedName: 'Verified Name',
    verifiedAt: 'Verified At',
    status: 'Status',
    active: 'Active',
    inactive: 'Inactive',
    // AI Settings
    aiEnabled: 'AI Auto-Response',
    aiEnabledDescription: 'Allow AI to automatically respond to customer messages',
    systemPrompt: 'System Prompt',
    systemPromptDescription: 'Instructions for how the AI should behave',
    responseLanguage: 'Response Language',
    arabic: 'Arabic',
    english: 'English',
    auto: 'Auto-detect',
    maxResponseLength: 'Max Response Length',
    responseDelay: 'Response Delay (seconds)',
    responseDelayDescription: 'Delay before AI responds (appears more natural)',
    // Notifications
    emailNotifications: 'Email Notifications',
    emailNotificationsDescription: 'Receive email alerts for new messages',
    notifyOnNewConversation: 'New Conversation',
    notifyOnNewConversationDescription: 'When a new customer starts a conversation',
    notifyOnInactivity: 'Inactivity Alert',
    notifyOnInactivityDescription: 'When a conversation has no response for too long',
    inactivityThreshold: 'Inactivity threshold (minutes)',
    // Business Hours
    businessHoursEnabled: 'Enable Business Hours',
    businessHoursDescription: 'AI responds differently outside business hours',
    outsideHoursMessage: 'Outside Hours Message',
    outsideHoursMessageDefault: 'Thank you for your message. Our business hours are Sunday-Thursday 9AM-6PM. We will respond during business hours.',
    // Actions
    save: 'Save Changes',
    saving: 'Saving...',
    saved: 'Changes saved',
    refresh: 'Refresh',
    noAccount: 'No WhatsApp account selected',
    selectAccount: 'Please select a WhatsApp account to configure',
  },
  ar: {
    title: 'إعدادات واتساب',
    description: 'تهيئة حساب واتساب للأعمال ووكيل الذكاء الاصطناعي',
    overview: 'نظرة عامة',
    aiSettings: 'وكيل الذكاء الاصطناعي',
    notifications: 'الإشعارات',
    businessHours: 'ساعات العمل',
    // Overview
    accountInfo: 'معلومات الحساب',
    phoneNumber: 'رقم الهاتف',
    verifiedName: 'الاسم المُوثّق',
    verifiedAt: 'تاريخ التوثيق',
    status: 'الحالة',
    active: 'نشط',
    inactive: 'غير نشط',
    // AI Settings
    aiEnabled: 'الرد التلقائي بالذكاء الاصطناعي',
    aiEnabledDescription: 'السماح للذكاء الاصطناعي بالرد تلقائيًا على رسائل العملاء',
    systemPrompt: 'تعليمات النظام',
    systemPromptDescription: 'تعليمات لسلوك الذكاء الاصطناعي',
    responseLanguage: 'لغة الرد',
    arabic: 'العربية',
    english: 'الإنجليزية',
    auto: 'اكتشاف تلقائي',
    maxResponseLength: 'الحد الأقصى لطول الرد',
    responseDelay: 'تأخير الرد (ثواني)',
    responseDelayDescription: 'تأخير قبل رد الذكاء الاصطناعي (يبدو أكثر طبيعية)',
    // Notifications
    emailNotifications: 'إشعارات البريد الإلكتروني',
    emailNotificationsDescription: 'استلام تنبيهات بالبريد للرسائل الجديدة',
    notifyOnNewConversation: 'محادثة جديدة',
    notifyOnNewConversationDescription: 'عندما يبدأ عميل جديد محادثة',
    notifyOnInactivity: 'تنبيه عدم النشاط',
    notifyOnInactivityDescription: 'عندما لا يكون هناك رد لفترة طويلة',
    inactivityThreshold: 'حد عدم النشاط (دقائق)',
    // Business Hours
    businessHoursEnabled: 'تفعيل ساعات العمل',
    businessHoursDescription: 'الذكاء الاصطناعي يرد بشكل مختلف خارج ساعات العمل',
    outsideHoursMessage: 'رسالة خارج ساعات العمل',
    outsideHoursMessageDefault: 'شكرًا على رسالتك. ساعات عملنا من الأحد إلى الخميس 9 صباحًا - 6 مساءً. سنرد خلال ساعات العمل.',
    // Actions
    save: 'حفظ التغييرات',
    saving: 'جاري الحفظ...',
    saved: 'تم حفظ التغييرات',
    refresh: 'تحديث',
    noAccount: 'لم يتم تحديد حساب واتساب',
    selectAccount: 'يرجى تحديد حساب واتساب لتهيئته',
  },
};

interface WhatsAppAccount {
  id: number;
  phone_number: string;
  phone_number_id: string;
  display_name: string;
  waba_id: string;
  is_active: boolean;
  verified_at: string;
  created_at: string;
}

// Backend response model
interface BackendAgentConfig {
  id: number;
  customer_id: number;
  business_name: string;
  tone: string;
  behavior_rules: string | null;
  custom_instructions: string | null;
  system_prompt: string;
  ai_enabled: boolean;
  ai_pause_until: string | null;
  ai_pause_duration_minutes: number;
  supported_language: 'english' | 'arabic' | 'both';
  max_response_length: number;
  response_delay_seconds: number;
  email_notifications: boolean;
  notify_on_new_conversation: boolean;
  notify_on_inactivity: boolean;
  inactivity_threshold_minutes: number;
  business_hours_enabled: boolean;
  outside_hours_message: string | null;
  created_at: string;
  updated_at: string;
}

// Frontend state model
interface AgentConfig {
  ai_enabled: boolean;
  system_prompt: string;
  response_language: 'ar' | 'en' | 'auto';
  max_response_length: number;
  response_delay_seconds: number;
  email_notifications: boolean;
  notify_on_new_conversation: boolean;
  notify_on_inactivity: boolean;
  inactivity_threshold_minutes: number;
  business_hours_enabled: boolean;
  outside_hours_message: string;
}

// Helper functions to map between frontend and backend formats
const mapBackendToFrontendLanguage = (lang: string): 'ar' | 'en' | 'auto' => {
  switch (lang) {
    case 'arabic': return 'ar';
    case 'english': return 'en';
    case 'both': return 'auto';
    default: return 'auto';
  }
};

const mapFrontendToBackendLanguage = (lang: string): string => {
  switch (lang) {
    case 'ar': return 'arabic';
    case 'en': return 'english';
    case 'auto': return 'both';
    default: return 'both';
  }
};

export default function WhatsAppSettings() {
  const [searchParams] = useSearchParams();
  const { language, isRTL } = useLanguage();
  const { toast } = useToast();
  const t = translations[language as keyof typeof translations] || translations.en;

  const accountId = searchParams.get('account');

  const [account, setAccount] = useState<WhatsAppAccount | null>(null);
  const [config, setConfig] = useState<AgentConfig>({
    ai_enabled: true,
    system_prompt: '',
    response_language: 'auto',
    max_response_length: 500,
    response_delay_seconds: 2,
    email_notifications: true,
    notify_on_new_conversation: true,
    notify_on_inactivity: false,
    inactivity_threshold_minutes: 30,
    business_hours_enabled: false,
    outside_hours_message: t.outsideHoursMessageDefault,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (accountId) {
      fetchAccountAndConfig();
    } else {
      setIsLoading(false);
    }
  }, [accountId]);

  const fetchAccountAndConfig = async () => {
    setIsLoading(true);
    try {
      const [accountRes, configRes] = await Promise.all([
        api.get<WhatsAppAccount>(`/api/v1/whatsapp/accounts/${accountId}`),
        api.get<BackendAgentConfig | null>(`/api/v1/whatsapp/accounts/${accountId}/config`),
      ]);
      setAccount(accountRes);
      // Map backend response to frontend state
      if (configRes) {
        setConfig({
          ai_enabled: configRes.ai_enabled,
          system_prompt: configRes.system_prompt || '',
          response_language: mapBackendToFrontendLanguage(configRes.supported_language),
          max_response_length: configRes.max_response_length,
          response_delay_seconds: configRes.response_delay_seconds,
          email_notifications: configRes.email_notifications,
          notify_on_new_conversation: configRes.notify_on_new_conversation,
          notify_on_inactivity: configRes.notify_on_inactivity,
          inactivity_threshold_minutes: configRes.inactivity_threshold_minutes,
          business_hours_enabled: configRes.business_hours_enabled,
          outside_hours_message: configRes.outside_hours_message || t.outsideHoursMessageDefault,
        });
      }
    } catch (error) {
      console.error('Failed to fetch settings:', error);
      toast({
        title: 'Error',
        description: 'Failed to load settings',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfigChange = (key: keyof AgentConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (!accountId) return;

    setIsSaving(true);
    try {
      // Map frontend state to backend request format
      await api.put(`/api/v1/whatsapp/accounts/${accountId}/config`, {
        ai_enabled: config.ai_enabled,
        system_prompt: config.system_prompt || null,
        response_language: config.response_language,
        max_response_length: config.max_response_length,
        response_delay_seconds: config.response_delay_seconds,
        email_notifications: config.email_notifications,
        notify_on_new_conversation: config.notify_on_new_conversation,
        notify_on_inactivity: config.notify_on_inactivity,
        inactivity_threshold_minutes: config.inactivity_threshold_minutes,
        business_hours_enabled: config.business_hours_enabled,
        outside_hours_message: config.outside_hours_message || null,
      });
      toast({
        title: 'Success',
        description: t.saved,
      });
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save settings:', error);
      toast({
        title: 'Error',
        description: 'Failed to save settings',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  if (!accountId) {
    return (
      <div className={`flex items-center justify-center min-h-[60vh] ${isRTL ? 'rtl' : 'ltr'}`}>
        <Card className="max-w-md w-full">
          <CardContent className="py-8 text-center">
            <Settings className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="font-medium text-lg">{t.noAccount}</h3>
            <p className="text-sm text-muted-foreground mt-2 mb-4">
              {t.selectAccount}
            </p>
            <Button asChild>
              <a href="/app/whatsapp">View WhatsApp Accounts</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{t.title}</h1>
          <p className="text-muted-foreground">{t.description}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={fetchAccountAndConfig}>
            <RefreshCw className="h-4 w-4 mr-2" />
            {t.refresh}
          </Button>
          <Button onClick={handleSave} disabled={!hasChanges || isSaving}>
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isSaving ? t.saving : t.save}
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">
            <MessageSquare className="h-4 w-4 mr-2" />
            {t.overview}
          </TabsTrigger>
          <TabsTrigger value="ai">
            <Bot className="h-4 w-4 mr-2" />
            {t.aiSettings}
          </TabsTrigger>
          <TabsTrigger value="notifications">
            <Bell className="h-4 w-4 mr-2" />
            {t.notifications}
          </TabsTrigger>
          <TabsTrigger value="hours">
            <Clock className="h-4 w-4 mr-2" />
            {t.businessHours}
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>{t.accountInfo}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground">{t.phoneNumber}</Label>
                  <p className="font-medium">{account?.phone_number || '-'}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{t.verifiedName}</Label>
                  <p className="font-medium">{account?.display_name || '-'}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{t.status}</Label>
                  <Badge variant={account?.is_active ? 'default' : 'secondary'}>
                    {account?.is_active ? t.active : t.inactive}
                  </Badge>
                </div>
                <div>
                  <Label className="text-muted-foreground">{t.verifiedAt}</Label>
                  <p className="font-medium">
                    {account?.verified_at ? new Date(account.verified_at).toLocaleDateString() : '-'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* AI Settings Tab */}
        <TabsContent value="ai">
          <Card>
            <CardHeader>
              <CardTitle>{t.aiSettings}</CardTitle>
              <CardDescription>{t.aiEnabledDescription}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label>{t.aiEnabled}</Label>
                  <p className="text-sm text-muted-foreground">{t.aiEnabledDescription}</p>
                </div>
                <Switch
                  checked={config.ai_enabled}
                  onCheckedChange={(checked) => handleConfigChange('ai_enabled', checked)}
                />
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>{t.systemPrompt}</Label>
                <p className="text-sm text-muted-foreground">{t.systemPromptDescription}</p>
                <Textarea
                  value={config.system_prompt}
                  onChange={(e) => handleConfigChange('system_prompt', e.target.value)}
                  rows={6}
                  placeholder="You are a helpful customer service agent for..."
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>{t.responseLanguage}</Label>
                  <Select
                    value={config.response_language}
                    onValueChange={(value) => handleConfigChange('response_language', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">{t.auto}</SelectItem>
                      <SelectItem value="ar">{t.arabic}</SelectItem>
                      <SelectItem value="en">{t.english}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>{t.maxResponseLength}</Label>
                  <Input
                    type="number"
                    value={config.max_response_length}
                    onChange={(e) => handleConfigChange('max_response_length', parseInt(e.target.value))}
                    min={100}
                    max={2000}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label>{t.responseDelay}</Label>
                <p className="text-sm text-muted-foreground">{t.responseDelayDescription}</p>
                <Input
                  type="number"
                  value={config.response_delay_seconds}
                  onChange={(e) => handleConfigChange('response_delay_seconds', parseInt(e.target.value))}
                  min={0}
                  max={10}
                  className="w-32"
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle>{t.notifications}</CardTitle>
              <CardDescription>{t.emailNotificationsDescription}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label>{t.emailNotifications}</Label>
                  <p className="text-sm text-muted-foreground">{t.emailNotificationsDescription}</p>
                </div>
                <Switch
                  checked={config.email_notifications}
                  onCheckedChange={(checked) => handleConfigChange('email_notifications', checked)}
                />
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div>
                  <Label>{t.notifyOnNewConversation}</Label>
                  <p className="text-sm text-muted-foreground">{t.notifyOnNewConversationDescription}</p>
                </div>
                <Switch
                  checked={config.notify_on_new_conversation}
                  onCheckedChange={(checked) => handleConfigChange('notify_on_new_conversation', checked)}
                  disabled={!config.email_notifications}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>{t.notifyOnInactivity}</Label>
                  <p className="text-sm text-muted-foreground">{t.notifyOnInactivityDescription}</p>
                </div>
                <Switch
                  checked={config.notify_on_inactivity}
                  onCheckedChange={(checked) => handleConfigChange('notify_on_inactivity', checked)}
                  disabled={!config.email_notifications}
                />
              </div>

              {config.notify_on_inactivity && (
                <div className="space-y-2 pl-4 border-l-2">
                  <Label>{t.inactivityThreshold}</Label>
                  <Input
                    type="number"
                    value={config.inactivity_threshold_minutes}
                    onChange={(e) => handleConfigChange('inactivity_threshold_minutes', parseInt(e.target.value))}
                    min={5}
                    max={180}
                    className="w-32"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Business Hours Tab */}
        <TabsContent value="hours">
          <Card>
            <CardHeader>
              <CardTitle>{t.businessHours}</CardTitle>
              <CardDescription>{t.businessHoursDescription}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label>{t.businessHoursEnabled}</Label>
                  <p className="text-sm text-muted-foreground">{t.businessHoursDescription}</p>
                </div>
                <Switch
                  checked={config.business_hours_enabled}
                  onCheckedChange={(checked) => handleConfigChange('business_hours_enabled', checked)}
                />
              </div>

              {config.business_hours_enabled && (
                <>
                  <Separator />
                  <div className="space-y-2">
                    <Label>{t.outsideHoursMessage}</Label>
                    <Textarea
                      value={config.outside_hours_message}
                      onChange={(e) => handleConfigChange('outside_hours_message', e.target.value)}
                      rows={4}
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
