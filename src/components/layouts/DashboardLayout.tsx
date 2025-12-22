/**
 * Dashboard Layout
 * 
 * Main layout for authenticated users with sidebar navigation.
 */

import { useState } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Home,
  Building2,
  Inbox,
  Bot,
  Settings,
  LogOut,
  Menu,
  X,
  ChevronDown,
  Users,
  BarChart3,
  CreditCard,
  BookOpen,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useAuth } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import { cn } from '@/lib/utils';
import logo from '@/assets/logo.png';

const translations = {
  en: {
    dashboard: 'Dashboard',
    home: 'Home',
    businesses: 'Businesses',
    inbox: 'Inbox',
    agentSetup: 'AI Agent',
    knowledgeBase: 'Knowledge Base',
    billing: 'Billing',
    settings: 'Settings',
    admin: 'Admin',
    customers: 'Customers',
    metrics: 'Metrics',
    users: 'Users',
    logout: 'Logout',
    profile: 'Profile',
  },
  ar: {
    dashboard: 'لوحة التحكم',
    home: 'الرئيسية',
    businesses: 'الأعمال',
    inbox: 'صندوق الوارد',
    agentSetup: 'وكيل الذكاء الاصطناعي',
    knowledgeBase: 'قاعدة المعرفة',
    billing: 'الفواتير',
    settings: 'الإعدادات',
    admin: 'الإدارة',
    customers: 'العملاء',
    metrics: 'المقاييس',
    users: 'المستخدمين',
    logout: 'تسجيل الخروج',
    profile: 'الملف الشخصي',
  },
};

interface NavItem {
  title: string;
  href: string;
  icon: React.ElementType;
  adminOnly?: boolean;
}

export default function DashboardLayout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const { user, logout, isAdmin } = useAuth();
  const { language, isRTL } = useLanguage();
  const location = useLocation();
  const navigate = useNavigate();

  const t = translations[language as keyof typeof translations] || translations.en;

  const mainNavItems: NavItem[] = [
    { title: t.home, href: '/app', icon: Home },
    { title: t.businesses, href: '/app/businesses', icon: Building2 },
    { title: t.inbox, href: '/app/inbox', icon: Inbox },
    { title: t.agentSetup, href: '/app/agent', icon: Bot },
    { title: t.knowledgeBase, href: '/app/knowledge-base', icon: BookOpen },
    { title: t.billing, href: '/app/billing', icon: CreditCard },
    { title: t.settings, href: '/app/settings', icon: Settings },
  ];

  const adminNavItems: NavItem[] = [
    { title: t.customers, href: '/app/admin/customers', icon: Users, adminOnly: true },
    { title: t.metrics, href: '/app/admin/metrics', icon: BarChart3, adminOnly: true },
    { title: t.users, href: '/app/admin/users', icon: Users, adminOnly: true },
  ];

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const NavLink = ({ item }: { item: NavItem }) => {
    const isActive = location.pathname === item.href;
    const Icon = item.icon;

    return (
      <Link
        to={item.href}
        onClick={() => setIsMobileMenuOpen(false)}
        className={cn(
          'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all',
          isActive
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
      >
        <Icon className="h-4 w-4" />
        {item.title}
      </Link>
    );
  };

  const SidebarContent = () => (
    <div className="flex h-full flex-col gap-2">
      {/* Logo */}
      <div className="flex h-14 items-center border-b px-4">
        <Link to="/" className="flex items-center gap-2">
          <img src={logo} alt="Genius AI" className="h-8" />
        </Link>
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 px-3">
        <nav className="flex flex-col gap-1 py-2">
          {mainNavItems.map((item) => (
            <NavLink key={item.href} item={item} />
          ))}

          {/* Admin Section */}
          {isAdmin && (
            <>
              <div className="my-2 px-3">
                <span className="text-xs font-semibold uppercase text-muted-foreground">
                  {t.admin}
                </span>
              </div>
              {adminNavItems.map((item) => (
                <NavLink key={item.href} item={item} />
              ))}
            </>
          )}
        </nav>
      </ScrollArea>

      {/* User Menu */}
      <div className="border-t p-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="w-full justify-start gap-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src={undefined} />
                <AvatarFallback>
                  {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                </AvatarFallback>
              </Avatar>
              <div className="flex flex-col items-start text-sm">
                <span className="font-medium">{user?.full_name || 'User'}</span>
                <span className="text-xs text-muted-foreground">{user?.email}</span>
              </div>
              <ChevronDown className="ml-auto h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>{t.profile}</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link to="/app/settings">{t.settings}</Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout} className="text-red-600">
              <LogOut className="mr-2 h-4 w-4" />
              {t.logout}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );

  return (
    <div className={`min-h-screen bg-background ${isRTL ? 'rtl' : 'ltr'}`}>
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-50 hidden w-64 border-r bg-card md:block">
        <SidebarContent />
      </aside>

      {/* Mobile Header */}
      <header className="sticky top-0 z-40 flex h-14 items-center gap-4 border-b bg-background px-4 md:hidden">
        <Sheet open={isMobileMenuOpen} onOpenChange={setIsMobileMenuOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon">
              <Menu className="h-6 w-6" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 p-0">
            <SidebarContent />
          </SheetContent>
        </Sheet>
        
        <Link to="/" className="flex items-center gap-2">
          <img src={logo} alt="Genius AI" className="h-8" />
        </Link>

        <div className="ml-auto">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <Avatar className="h-8 w-8">
                  <AvatarFallback>
                    {user?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>{user?.email}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link to="/app/settings">{t.settings}</Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout} className="text-red-600">
                <LogOut className="mr-2 h-4 w-4" />
                {t.logout}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Main Content */}
      <main className="md:pl-64">
        <div className="container mx-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
