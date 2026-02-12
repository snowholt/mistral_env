import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { LanguageProvider } from "@/hooks/useLanguage";
import { LanguageProvider as LandingLanguageProvider } from "@/contexts/LanguageContext";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { AuthProvider } from "@/hooks/useAuth";
import { ProtectedRoute, PublicRoute } from "@/components/ProtectedRoute";
import ChatWidget from "@/components/ChatWidget";

// Public pages
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Policy from "./pages/privacy-policy";
import Term from "./pages/terms";
import RequestDemo from "./pages/RequestDemo";

// Auth pages
import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import ForgotPassword from "./pages/auth/ForgotPassword";
import ResetPassword from "./pages/auth/ResetPassword";
import GuestLogin from "./pages/auth/GuestLogin";
import VerifyEmail from "./pages/auth/VerifyEmail";

// Dashboard pages
import DashboardLayout from "@/components/layouts/DashboardLayout";
import DashboardHome from "./pages/app/Home";
import Businesses from "./pages/app/Businesses";
import Inbox from "./pages/app/Inbox";
import AgentSetup from "./pages/app/AgentSetup";
import AgentSetupWizard from "./pages/app/AgentSetupWizard";
import KnowledgeBase from "./pages/app/KnowledgeBase";
import Billing from "./pages/app/Billing";
import Settings from "./pages/app/Settings";
import VoiceDemo from "./pages/app/VoiceDemo";
import WhatsAppConnect from "./pages/app/WhatsAppConnect";
import WhatsAppSettings from "./pages/app/WhatsAppSettings";

// Admin pages
import AdminCustomers from "./pages/app/admin/Customers";
import AdminMetrics from "./pages/app/admin/Metrics";
import AdminUsers from "./pages/app/admin/Users";
import AdminDemoRequests from "./pages/app/admin/DemoRequests";

const queryClient = new QueryClient();

// Chat widget wrapper - only shows on public pages
function PublicChatWidget() {
  const location = useLocation();
  const isPublicPage = !location.pathname.startsWith('/app');

  if (!isPublicPage) return null;

  return (
    <ChatWidget
      widgetToken="wt_uf7GDPwuw4vBwGYed26_0-yepA32XwjUbjZas1GnRkI"
      apiUrl={import.meta.env.VITE_API_URL || 'https://api.gmai.sa'}
      primaryColor="#00b3a4"
      headerText="مساعد الذكاء الاصطناعي"
      placeholderText="اكتب رسالتك..."
      welcomeMessage="مرحباً! 👋 كيف يمكنني مساعدتك اليوم؟"
      position="bottom-right"
    />
  );
}

// Inner app component with access to router context
function AppContent() {
  return (
    <>
      <Routes>
        {/* Public Landing Pages */}
        <Route path="/" element={<Index />} />
        <Route path="/privacy-policy" element={<Policy />} />
        <Route path="/terms" element={<Term />} />
        <Route path="/request-demo" element={<RequestDemo />} />

        {/* Auth Pages - Redirect to dashboard if logged in */}
        <Route path="/login" element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        } />
        <Route path="/register" element={
          <PublicRoute>
            <Register />
          </PublicRoute>
        } />
        <Route path="/demo/login" element={<GuestLogin />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/auth/verify-email" element={<VerifyEmail />} />

        {/* Protected Dashboard Routes */}
        <Route path="/app" element={
          <ProtectedRoute>
            <DashboardLayout />
          </ProtectedRoute>
        }>
          <Route index element={<DashboardHome />} />
          <Route path="businesses" element={<Businesses />} />
          <Route path="inbox" element={<Inbox />} />
          <Route path="whatsapp" element={<WhatsAppConnect />} />
          <Route path="whatsapp/settings" element={<WhatsAppSettings />} />
          <Route path="agent" element={<AgentSetup />} />
          <Route path="agent-wizard" element={<AgentSetupWizard />} />
          <Route path="knowledge-base" element={<KnowledgeBase />} />
          <Route path="billing" element={<Billing />} />
          <Route path="settings" element={<Settings />} />
          <Route path="demo" element={<VoiceDemo />} />

          {/* Admin Routes - Requires admin role */}
          <Route path="admin/customers" element={
            <ProtectedRoute requireAdmin>
              <AdminCustomers />
            </ProtectedRoute>
          } />
          <Route path="admin/demo-requests" element={
            <ProtectedRoute requireAdmin>
              <AdminDemoRequests />
            </ProtectedRoute>
          } />
          <Route path="admin/metrics" element={
            <ProtectedRoute requireAdmin>
              <AdminMetrics />
            </ProtectedRoute>
          } />
          <Route path="admin/users" element={
            <ProtectedRoute requireAdmin>
              <AdminUsers />
            </ProtectedRoute>
          } />
        </Route>

        {/* 404 Catch-all */}
        <Route path="*" element={<NotFound />} />
      </Routes>
      <PublicChatWidget />
    </>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <LandingLanguageProvider>
        <TooltipProvider>
          <LanguageProvider>
            <AuthProvider>
              <Toaster />
              <Sonner />
              <BrowserRouter>
                <AppContent />
              </BrowserRouter>
            </AuthProvider>
          </LanguageProvider>
        </TooltipProvider>
      </LandingLanguageProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;

