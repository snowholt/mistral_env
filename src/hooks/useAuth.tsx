/**
 * Authentication Context and Hook
 * 
 * Provides authentication state and methods throughout the app.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authApi, tokenManager, User, ApiError } from '@/lib/api';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  isAdmin: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<{ message: string }>;
  logout: () => void;
  verifyEmail: (token: string) => Promise<void>;
  forgotPassword: (email: string) => Promise<void>;
  resetPassword: (token: string, newPassword: string) => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    isAdmin: false,
  });

  // Initialize auth state from storage
  useEffect(() => {
    const initAuth = async () => {
      const token = tokenManager.getAccessToken();
      const storedUser = tokenManager.getUser();

      if (token) {
        try {
          // Verify token is still valid by fetching user
          const user = await authApi.getMe();
          setState({
            user,
            isAuthenticated: true,
            isLoading: false,
            isAdmin: user.role === 'admin',
          });
        } catch {
          // Token is invalid, clear it
          tokenManager.clearTokens();
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            isAdmin: false,
          });
        }
      } else if (storedUser) {
        // Have cached user but no token - clear state
        tokenManager.clearTokens();
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          isAdmin: false,
        });
      } else {
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          isAdmin: false,
        });
      }
    };

    initAuth();
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await authApi.login(email, password);
    setState({
      user: response.user,
      isAuthenticated: true,
      isLoading: false,
      isAdmin: response.user.role === 'admin',
    });
  }, []);

  const register = useCallback(async (email: string, password: string, fullName?: string) => {
    const response = await authApi.register(email, password, fullName);
    return response;
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    setState({
      user: null,
      isAuthenticated: false,
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
    } catch {
      // Silently fail, token might be expired
    }
  }, []);

  const value: AuthContextType = {
    ...state,
    login,
    register,
    logout,
    verifyEmail,
    forgotPassword,
    resetPassword,
    refreshUser,
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
