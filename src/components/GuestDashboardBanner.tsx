import { useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useLanguage } from "@/hooks/useLanguage";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Clock, MessageSquare, TrendingUp, AlertCircle } from "lucide-react";
import { Link } from "react-router-dom";

export default function GuestDashboardBanner() {
  const { guestUser, refreshGuestUser } = useAuth();
  const { language } = useLanguage();

  useEffect(() => {
    // Refresh guest user data on mount
    refreshGuestUser();
  }, [refreshGuestUser]);

  if (!guestUser) return null;

  const conversationProgress = (guestUser.conversations_used / guestUser.max_conversations) * 100;
  const isExpiringSoon = guestUser.days_remaining <= 2;
  const isLimitReached = guestUser.is_limit_reached;
  const isExpired = guestUser.is_expired;

  // Determine alert variant
  const getAlertVariant = () => {
    if (isExpired || isLimitReached) return "destructive";
    if (isExpiringSoon || conversationProgress >= 80) return "default";
    return "default";
  };

  return (
    <Alert variant={getAlertVariant()} className="mb-6">
      <Sparkles className="h-4 w-4" />
      <AlertTitle className="flex items-center justify-between">
        <span className="font-semibold">
          {language === 'ar' ? 'حساب تجريبي' : 'Demo Account'}
        </span>
        <Badge variant={guestUser.can_access ? "default" : "secondary"}>
          {guestUser.can_access
            ? (language === 'ar' ? 'نشط' : 'Active')
            : isExpired
              ? (language === 'ar' ? 'منتهي' : 'Expired')
              : (language === 'ar' ? 'محدود' : 'Limited')
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
                  <span className="font-medium">
                    {language === 'ar' ? 'المحادثات' : 'Conversations'}
                  </span>
                </div>
                <span className="font-bold">
                  {guestUser.conversations_used} / {guestUser.max_conversations}
                </span>
              </div>
              <Progress value={conversationProgress} className="h-2" />
              <p className="text-xs text-muted-foreground">
                {guestUser.conversations_remaining > 0
                  ? (language === 'ar'
                    ? `${guestUser.conversations_remaining} محادثة متبقية`
                    : `${guestUser.conversations_remaining} conversations remaining`)
                  : (language === 'ar' ? 'لا توجد محادثات متبقية' : 'No conversations remaining')
                }
              </p>
            </div>

            {/* Time Remaining */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  <span className="font-medium">
                    {language === 'ar' ? 'الوقت المتبقي' : 'Time Remaining'}
                  </span>
                </div>
                <span className="font-bold">
                  {guestUser.days_remaining} {language === 'ar' ? 'يوم' : 'days'}
                </span>
              </div>
              <Progress 
                value={(guestUser.days_remaining / 7) * 100} 
                className="h-2" 
              />
              <p className="text-xs text-muted-foreground">
                {isExpiringSoon && !isExpired
                  ? (language === 'ar'
                    ? '⚠️ ينتهي قريباً!'
                    : '⚠️ Expiring soon!')
                  : isExpired
                    ? (language === 'ar' ? '❌ منتهي الصلاحية' : '❌ Expired')
                    : (language === 'ar'
                      ? `تنتهي في ${new Date(guestUser.expires_at).toLocaleDateString('ar')}`
                      : `Expires on ${new Date(guestUser.expires_at).toLocaleDateString('en')}`)
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
                  {language === 'ar' 
                    ? 'انتهت صلاحية الوصول التجريبي' 
                    : 'Demo access has expired'
                  }
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {isExpired
                    ? (language === 'ar'
                      ? 'انتهت فترة التجربة المجانية'
                      : 'Your free trial period has ended')
                    : (language === 'ar'
                      ? 'وصلت إلى الحد الأقصى للمحادثات'
                      : 'You have reached the maximum conversations limit')
                  }
                </p>
              </div>
            </div>
          )}

          {/* Upgrade CTA */}
          <div className="flex items-center justify-between pt-2 border-t">
            <p className="text-sm">
              {language === 'ar'
                ? 'استمتع بمزايا غير محدودة مع الخطة الكاملة'
                : 'Enjoy unlimited features with the full plan'
              }
            </p>
            <Link to="/app/billing">
              <Button size="sm" variant="default" className="gap-2">
                <TrendingUp className="h-4 w-4" />
                {language === 'ar' ? 'ترقية الحساب' : 'Upgrade Now'}
              </Button>
            </Link>
          </div>
        </div>
      </AlertDescription>
    </Alert>
  );
}
