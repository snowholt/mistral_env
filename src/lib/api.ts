/**
 * API Client for BeautyAI Backend
 * 
 * Handles all HTTP requests with JWT authentication,
 * token refresh, and error handling.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://api.gmai.sa';

// Token storage keys
const ACCESS_TOKEN_KEY = 'beautyai_access_token';
const REFRESH_TOKEN_KEY = 'beautyai_refresh_token';
const USER_KEY = 'beautyai_user';

// Types
export interface User {
  id: number;
  email: string;
  full_name: string | null;
  role: 'user' | 'admin';
  is_verified: boolean;
  created_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token?: string;
  token_type: string;
}

export interface ApiError {
  detail: string;
  status: number;
}

// Token management
export const tokenManager = {
  getAccessToken: (): string | null => {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  },

  getRefreshToken: (): string | null => {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  },

  setTokens: (tokens: AuthTokens): void => {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
    if (tokens.refresh_token) {
      localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
    }
  },

  clearTokens: (): void => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  },

  getUser: (): User | null => {
    const userJson = localStorage.getItem(USER_KEY);
    if (userJson) {
      try {
        return JSON.parse(userJson);
      } catch {
        return null;
      }
    }
    return null;
  },

  setUser: (user: User): void => {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
};

// API Client class
class ApiClient {
  private baseUrl: string;
  private isRefreshing: boolean = false;
  private refreshPromise: Promise<boolean> | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async getHeaders(includeAuth: boolean = true): Promise<HeadersInit> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (includeAuth) {
      const token = tokenManager.getAccessToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    }

    return headers;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (response.ok) {
      // Handle empty responses
      const text = await response.text();
      if (!text) return {} as T;
      return JSON.parse(text);
    }

    // Handle errors
    let errorDetail = 'An error occurred';
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorData.message || errorDetail;
    } catch {
      errorDetail = response.statusText;
    }

    const error: ApiError = {
      detail: errorDetail,
      status: response.status,
    };

    throw error;
  }

  private async refreshTokenIfNeeded(): Promise<boolean> {
    // If already refreshing, wait for that to complete
    if (this.isRefreshing && this.refreshPromise) {
      return this.refreshPromise;
    }

    const refreshToken = tokenManager.getRefreshToken();
    if (!refreshToken) {
      return false;
    }

    this.isRefreshing = true;
    this.refreshPromise = this.doRefreshToken(refreshToken);

    try {
      const success = await this.refreshPromise;
      return success;
    } finally {
      this.isRefreshing = false;
      this.refreshPromise = null;
    }
  }

  private async doRefreshToken(refreshToken: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/whatsapp/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (response.ok) {
        const data = await response.json();
        tokenManager.setTokens(data);
        return true;
      }

      // Refresh failed, clear tokens
      tokenManager.clearTokens();
      return false;
    } catch {
      tokenManager.clearTokens();
      return false;
    }
  }

  async request<T>(
    endpoint: string,
    options: RequestInit = {},
    includeAuth: boolean = true,
    retryOnAuthError: boolean = true,
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = await this.getHeaders(includeAuth);

    const response = await fetch(url, {
      ...options,
      headers: {
        ...headers,
        ...(options.headers || {}),
      },
    });

    // Handle 401 Unauthorized - try to refresh token
    if (response.status === 401 && includeAuth && retryOnAuthError) {
      const refreshed = await this.refreshTokenIfNeeded();
      if (refreshed) {
        // Retry the request with new token
        return this.request<T>(endpoint, options, includeAuth, false);
      }
      // Refresh failed, throw error
      throw { detail: 'Session expired. Please log in again.', status: 401 };
    }

    return this.handleResponse<T>(response);
  }

  // Convenience methods
  async get<T>(endpoint: string, includeAuth: boolean = true): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' }, includeAuth);
  }

  async post<T>(endpoint: string, data?: unknown, includeAuth: boolean = true): Promise<T> {
    return this.request<T>(
      endpoint,
      {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
      },
      includeAuth,
    );
  }

  async put<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async patch<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  // File upload
  async upload<T>(endpoint: string, formData: FormData): Promise<T> {
    const token = tokenManager.getAccessToken();
    const headers: HeadersInit = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    // Don't set Content-Type for FormData, browser will set it with boundary

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers,
      body: formData,
    });

    return this.handleResponse<T>(response);
  }
}

// Export singleton instance
export const api = new ApiClient();

// Auth-specific API calls
export const authApi = {
  login: async (email: string, password: string) => {
    const response = await api.post<AuthTokens & { user: User }>(
      '/api/v1/whatsapp/auth/login',
      { email, password },
      false,
    );
    tokenManager.setTokens(response);
    tokenManager.setUser(response.user);
    return response;
  },

  register: async (email: string, password: string, fullName?: string) => {
    const response = await api.post<{ message: string; user_id: number }>(
      '/api/v1/whatsapp/auth/register',
      { email, password, full_name: fullName },
      false,
    );
    return response;
  },

  verifyEmail: async (token: string) => {
    return api.post<{ message: string }>(
      '/api/v1/whatsapp/auth/verify-email',
      { token },
      false,
    );
  },

  resendVerification: async (email: string) => {
    return api.post<{ message: string }>(
      '/api/v1/whatsapp/auth/resend-verification',
      { email },
      false,
    );
  },

  forgotPassword: async (email: string) => {
    return api.post<{ message: string }>(
      '/api/v1/whatsapp/auth/forgot-password',
      { email },
      false,
    );
  },

  resetPassword: async (token: string, newPassword: string) => {
    return api.post<{ message: string }>(
      '/api/v1/whatsapp/auth/reset-password',
      { token, new_password: newPassword },
      false,
    );
  },

  getMe: async () => {
    const user = await api.get<User>('/api/v1/whatsapp/auth/me');
    tokenManager.setUser(user);
    return user;
  },

  logout: () => {
    tokenManager.clearTokens();
  },
};

export default api;
