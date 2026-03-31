/**
 * Authentication Context and Hook
 * 
 * Provides authentication state and methods throughout the app.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authApi, tokenManager, guestApi, User, ApiError, AUTH_EVENTS } from '@/lib/api';

interface GuestUser {
  id: number;
  email: string;
  is_active: boolean;
  max_conversations: number;
  conversations_used: number;
  expires_at: string;
  is_expired: boolean;
  is_limit_reached: boolean;
  can_access: boolean;
  days_remaining: number;
  conversations_remaining: number;
}

interface AuthState {
  user: User | null;
  guestUser: GuestUser | null;
  isAuthenticated: boolean;
  isGuest: boolean;
  isLoading: boolean;
  isAdmin: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  guestLogin: (accessToken: string) => Promise<void>;
  guestPasswordLogin: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<{ message: string }>;
  logout: () => void;
  verifyEmail: (token: string) => Promise<void>;
  forgotPassword: (email: string) => Promise<void>;
  resetPassword: (token: string, newPassword: string) => Promise<void>;
  refreshUser: () => Promise<void>;
  refreshGuestUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    guestUser: null,
    isAuthenticated: false,
    isGuest: false,
    isLoading: true,
    isAdmin: false,
  });

  // Initialize auth state from storage
  useEffect(() => {
    const initAuth = async () => {
      const token = tokenManager.getAccessToken();
      const storedUser = tokenManager.getUser();
      const isGuestStored = localStorage.getItem('isGuest') === 'true';

      if (token) {
        try {
          // Check if this is a guest user
          if (isGuestStored) {
            const guestUser = await guestApi.getProfile();
            setState({
              user: null,
              guestUser,
              isAuthenticated: true,
              isGuest: true,
              isLoading: false,
              isAdmin: false,
            });
          } else {
            // Regular user - verify token is still valid by fetching user
            const user = await authApi.getMe();
            setState({
              user,
              guestUser: null,
              isAuthenticated: true,
              isGuest: false,
              isLoading: false,
              isAdmin: user.role === 'admin',
            });
          }
        } catch {
          // Token is invalid, clear it
          tokenManager.clearTokens();
          localStorage.removeItem('isGuest');
          setState({
            user: null,
            guestUser: null,
            isAuthenticated: false,
            isGuest: false,
            isLoading: false,
            isAdmin: false,
          });
        }
      } else if (storedUser) {
        // Have cached user but no token - clear state
        tokenManager.clearTokens();
        localStorage.removeItem('isGuest');
        setState({
          user: null,
          guestUser: null,
          isAuthenticated: false,
          isGuest: false,
          isLoading: false,
          isAdmin: false,
        });
      } else {
        setState({
          user: null,
          guestUser: null,
          isAuthenticated: false,
          isGuest: false,
          isLoading: false,
          isAdmin: false,
        });
      }
    };

    initAuth();
  }, []);

  useEffect(() => {
    const handleSessionExpired = () => {
      tokenManager.clearTokens();
      localStorage.removeItem('isGuest');
      setState({
        user: null,
        guestUser: null,
        isAuthenticated: false,
        isGuest: false,
        isLoading: false,
        isAdmin: false,
      });
    };

    window.addEventListener(AUTH_EVENTS.SESSION_EXPIRED, handleSessionExpired as EventListener);
    return () => {
      window.removeEventListener(AUTH_EVENTS.SESSION_EXPIRED, handleSessionExpired as EventListener);
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await authApi.login(email, password);
    localStorage.removeItem('isGuest'); // Clear guest flag
    setState({
      user: response.user,
      guestUser: null,
      isAuthenticated: true,
      isGuest: false,
      isLoading: false,
      isAdmin: response.user.role === 'admin',
    });
  }, []);

  const guestLogin = useCallback(async (accessToken: string) => {
    const response = await guestApi.login(accessToken);
    localStorage.setItem('isGuest', 'true');
    setState({
      user: null,
      guestUser: response.guest_user,
      isAuthenticated: true,
      isGuest: true,
      isLoading: false,
      isAdmin: false,
    });
  }, []);

  const guestPasswordLogin = useCallback(async (email: string, password: string) => {
    const response = await guestApi.passwordLogin(email, password);
    localStorage.setItem('isGuest', 'true');
    setState({
      user: null,
      guestUser: response.guest_user,
      isAuthenticated: true,
      isGuest: true,
      isLoading: false,
      isAdmin: false,
    });
  }, []);

  const register = useCallback(async (email: string, password: string, fullName?: string) => {
    const response = await authApi.register(email, password, fullName);
    return response;
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    localStorage.removeItem('isGuest');
    setState({
      user: null,
      guestUser: null,
      isAuthenticated: false,
      isGuest: false,
      isLoading: false,
      isAdmin: false,
    });
  }, []);

  const verifyEmail = useCallback(async (token: string) => {
    await authApi.verifyEmail(token);
  }, []);

  const forgotPassword = useCallback(async (email: string) => {
    await authApi.forgotPassword(email);
  }, []);

  const resetPassword = useCallback(async (token: string, newPassword: string) => {
    await authApi.resetPassword(token, newPassword);
  }, []);

  const refreshUser = useCallback(async () => {
    try {
      const user = await authApi.getMe();
      setState(prev => ({
        ...prev,
        user,
        isAdmin: user.role === 'admin',
      }));
      return user;  // Return refreshed user for immediate use
    } catch {
      // Silently fail, token might be expired
      return null;
    }
  }, []);

  const refreshGuestUser = useCallback(async () => {
    try {
      const guestUser = await guestApi.getProfile();
      setState(prev => ({
        ...prev,
        guestUser,
      }));
    } catch {
      // Silently fail, token might be expired
    }
  }, []);

  const value: AuthContextType = {
    ...state,
    login,
    guestLogin,
    guestPasswordLogin,
    register,
    logout,
    verifyEmail,
    forgotPassword,
    resetPassword,
    refreshUser,
    refreshGuestUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Type guard for API errors
export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'detail' in error &&
    'status' in error
  );
}

export default AuthContext;
