import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { LanguageProvider } from "@/hooks/useLanguage";
import { AuthProvider } from "@/hooks/useAuth";
import { ProtectedRoute, PublicRoute } from "@/components/ProtectedRoute";

// Public pages
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Policy from "./pages/privacy-policy";
import Term from "./pages/terms";

// Auth pages
import Login from "./pages/auth/Login";
import Register from "./pages/auth/Register";
import ForgotPassword from "./pages/auth/ForgotPassword";
import ResetPassword from "./pages/auth/ResetPassword";

// Dashboard pages
import DashboardLayout from "@/components/layouts/DashboardLayout";
import DashboardHome from "./pages/app/Home";

// Admin pages
import AdminCustomers from "./pages/app/admin/Customers";
import AdminMetrics from "./pages/app/admin/Metrics";
import AdminUsers from "./pages/app/admin/Users";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <LanguageProvider>
        <AuthProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              {/* Public Landing Pages */}
              <Route path="/" element={<Index />} />
              <Route path="/privacy-policy" element={<Policy />} />
              <Route path="/terms" element={<Term />} />

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
              <Route path="/forgot-password" element={<ForgotPassword />} />
              <Route path="/reset-password" element={<ResetPassword />} />

              {/* Protected Dashboard Routes */}
              <Route path="/app" element={
                <ProtectedRoute>
                  <DashboardLayout />
                </ProtectedRoute>
              }>
                <Route index element={<DashboardHome />} />
                {/* Add more dashboard routes here */}
                {/* <Route path="businesses" element={<Businesses />} /> */}
                {/* <Route path="inbox" element={<Inbox />} /> */}
                {/* <Route path="agent" element={<AgentSetup />} /> */}
                {/* <Route path="knowledge-base" element={<KnowledgeBase />} /> */}
                {/* <Route path="billing" element={<Billing />} /> */}
                {/* <Route path="settings" element={<Settings />} /> */}
                
                {/* Admin Routes - Requires admin role */}
                <Route path="admin/customers" element={
                  <ProtectedRoute requireAdmin>
                    <AdminCustomers />
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
          </BrowserRouter>
        </AuthProvider>
      </LanguageProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
