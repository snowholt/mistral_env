/**
 * API Client for BeautyAI Backend
 * 
 * Handles all HTTP requests with JWT authentication,
 * token refresh, and error handling.
 */

// Prefer same-origin API calls by default so portal.gmai.sa can proxy /api/* to the backend.
// Override with VITE_API_URL if you want a dedicated API origin.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Token storage keys
const ACCESS_TOKEN_KEY = 'beautyai_access_token';
const REFRESH_TOKEN_KEY = 'beautyai_refresh_token';
const USER_KEY = 'beautyai_user';
const GUEST_TOKEN_KEY = 'beautyai_guest_token';

// Types
export interface User {
  id: number;
  email: string;
  full_name: string | null;
  role: 'user' | 'admin' | 'guest';
  is_verified: boolean;
  created_at: string;
  // Guest-specific fields (only populated when role=guest)
  expires_at?: string;
  max_conversations?: number;
  conversations_used?: number;
  is_expired?: boolean;
  is_limit_reached?: boolean;
  days_remaining?: number;
  conversations_remaining?: number;
  can_access_demo?: boolean;
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

  setToken: (token: string): void => {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
  },

  getGuestToken: (): string | null => {
    return localStorage.getItem(GUEST_TOKEN_KEY);
  },

  setGuestToken: (token: string): void => {
    localStorage.setItem(GUEST_TOKEN_KEY, token);
  },

  clearTokens: (): void => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(GUEST_TOKEN_KEY);
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
      const response = await fetch(`${this.baseUrl}/api/v1/auth/refresh`, {
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
      '/api/v1/auth/login',
      { email, password },
      false,
    );
    tokenManager.setTokens(response);
    tokenManager.setUser(response.user);
    return response;
  },

  register: async (email: string, password: string, fullName?: string) => {
    const response = await api.post<{ message: string; user_id: number }>(
      '/api/v1/auth/register',
      { email, password, full_name: fullName },
      false,
    );
    return response;
  },

  verifyEmail: async (token: string) => {
    return api.post<{ message: string }>(
      '/api/v1/auth/verify-email',
      { token },
      false,
    );
  },

  resendVerification: async (email: string) => {
    return api.post<{ message: string }>(
      '/api/v1/auth/resend-verification',
      { email },
      false,
    );
  },

  forgotPassword: async (email: string) => {
    return api.post<{ message: string }>(
      '/api/v1/auth/forgot-password',
      { email },
      false,
    );
  },

  resetPassword: async (token: string, newPassword: string) => {
    return api.post<{ message: string }>(
      '/api/v1/auth/reset-password',
      { token, new_password: newPassword },
      false,
    );
  },

  getMe: async () => {
    const user = await api.get<User>('/api/v1/auth/me');
    tokenManager.setUser(user);
    return user;
  },

  logout: () => {
    tokenManager.clearTokens();
  },
};

// ===== Demo Request API =====
export const demoApi = {
  submitDemoRequest: async (data: {
    first_name: string;
    last_name: string;
    email: string;
    phone?: string;
    company?: string;
    company_size?: string;
    message?: string;
  }) => {
    return api.post<{ message: string; request_id: number }>(
      '/api/v1/demo-requests',
      data,
      false,
    );
  },
};

// ===== Guest Auth API =====
export const guestApi = {
  // Legacy token-based login (backward compatibility)
  login: async (accessToken: string) => {
    const response = await api.post<{
      jwt_token: string;
      access_token: string; // Keep for backward compatibility types if needed
      token_type: string;
      guest_user: {
        id: number;
        email: string;
        access_token: string;
        is_active: boolean;
        is_activated: boolean;
        max_conversations: number;
        conversations_used: number;
        expires_at: string;
      };
    }>(
      '/api/v1/auth/guest/login',
      { access_token: accessToken },
      false,
    );
    // Store JWT token for API access
    tokenManager.setToken(response.jwt_token);
    // Store Guest Access Token for guest-specific endpoints
    tokenManager.setGuestToken(response.guest_user.access_token);
    return response;
  },

  // New password-based login for activated accounts
  passwordLogin: async (email: string, password: string) => {
    const response = await api.post<{
      jwt_token: string;
      access_token: string;
      token_type: string;
      guest_user: {
        id: number;
        email: string;
        access_token: string;
        is_active: boolean;
        is_activated: boolean;
        max_conversations: number;
        conversations_used: number;
        expires_at: string;
      };
    }>(
      '/api/v1/auth/guest/login',
      { email, password },
      false,
    );
    // Store JWT token for API access
    tokenManager.setToken(response.jwt_token);
    // Store Guest Access Token for guest-specific endpoints
    tokenManager.setGuestToken(response.guest_user.access_token);
    return response;
  },

  // Validate setup token from email link
  validateSetupToken: async (token: string) => {
    return api.post<{
      valid: boolean;
      email: string;
      message: string;
    }>(
      '/api/v1/auth/guest/validate-setup-token',
      { token },
      false,
    );
  },

  // Set password for guest account (activates the account)
  setPassword: async (token: string, password: string) => {
    return api.post<{
      message: string;
      email: string;
      is_activated: boolean;
    }>(
      '/api/v1/auth/guest/set-password',
      { token, password },
      false,
    );
  },

  // Get password requirements for display
  getPasswordRequirements: async () => {
    return api.get<{
      min_length: number;
      require_uppercase: boolean;
      require_lowercase: boolean;
      require_digit: boolean;
      require_special: boolean;
      special_characters: string;
    }>(
      '/api/v1/auth/guest/password-requirements',
      false,
    );
  },

  getProfile: async () => {
    const token = tokenManager.getGuestToken();
    return api.get<{
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
    }>(token ? `/api/v1/auth/guest/me?token=${encodeURIComponent(token)}` : `/api/v1/auth/guest/me`);
  },

  validateAccess: async () => {
    const token = tokenManager.getGuestToken();
    if (!token) {
      console.warn('[GuestAuth] No guest token available for validate-access');
      return {
        can_access: false,
        is_expired: false,
        is_limit_reached: false,
        days_remaining: 0,
        conversations_remaining: 0,
        message: 'No guest token available'
      };
    }
    return api.get<{
      can_access: boolean;
      is_expired: boolean;
      is_limit_reached: boolean;
      days_remaining: number;
      conversations_remaining: number;
      message: string;
    }>(`/api/v1/auth/guest/validate-access?token=${token}`);
  },

  incrementUsage: async () => {
    const token = tokenManager.getGuestToken();
    if (!token) {
      console.warn('[GuestAuth] No guest token available for increment-usage');
      return { 
        message: 'No guest token available',
        conversations_used: 0,
        conversations_remaining: 0 
      };
    }
    return api.post<{
      message: string;
      conversations_used: number;
      conversations_remaining: number;
    }>(`/api/v1/auth/guest/increment-usage?token=${token}`, {});
  },

  logout: () => {
    tokenManager.clearTokens();
  },
};

// ===== Admin Demo Request API =====
export const adminDemoApi = {
  listDemoRequests: async (params?: {
    status?: 'pending' | 'approved' | 'rejected';
    skip?: number;
    limit?: number;
  }) => {
    const queryParams = new URLSearchParams();
    if (params?.status) queryParams.append('status', params.status);
    if (params?.skip) queryParams.append('skip', params.skip.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const url = `/api/v1/admin/demo-requests${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return api.get<{
      total: number;
      items: Array<{
        id: number;
        first_name: string;
        last_name: string;
        email: string;
        phone: string | null;
        company: string | null;
        company_size: string | null;
        message: string | null;
        status: 'pending' | 'approved' | 'rejected';
        created_at: string;
        updated_at: string;
        admin_notes: string | null;
        assigned_to_admin_id: number | null;
        scheduled_follow_up: string | null;
      }>;
    }>(url);
  },

  getDemoRequest: async (id: number) => {
    return api.get<{
      id: number;
      first_name: string;
      last_name: string;
      email: string;
      phone: string | null;
      company: string | null;
      company_size: string | null;
      message: string | null;
      status: 'pending' | 'approved' | 'rejected';
      created_at: string;
      updated_at: string;
      admin_notes: string | null;
      assigned_to_admin_id: number | null;
      scheduled_follow_up: string | null;
      assigned_to?: { id: number; email: string; full_name: string };
    }>(`/api/v1/admin/demo-requests/${id}`);
  },

  approveDemoRequest: async (
    id: number,
    data: {
      max_conversations?: number;
      days_valid?: number;
      admin_notes?: string;
    },
  ) => {
    return api.patch<{
      message: string;
      demo_request: any;
      guest_user: any;
    }>(`/api/v1/admin/demo-requests/${id}/approve`, data);
  },

  rejectDemoRequest: async (id: number, admin_notes?: string) => {
    return api.patch<{ message: string; demo_request: any }>(
      `/api/v1/admin/demo-requests/${id}/reject`,
      { admin_notes },
    );
  },

  updateDemoRequest: async (
    id: number,
    data: {
      admin_notes?: string;
      assigned_to_admin_id?: number | null;
      scheduled_follow_up?: string | null;
    },
  ) => {
    return api.patch<{ message: string; demo_request: any }>(
      `/api/v1/admin/demo-requests/${id}`,
      data,
    );
  },

  deleteDemoRequest: async (id: number) => {
    return api.delete<{ message: string }>(
      `/api/v1/admin/demo-requests/${id}`,
    );
  },

  listGuestUsers: async (params?: { skip?: number; limit?: number }) => {
    const queryParams = new URLSearchParams();
    if (params?.skip) queryParams.append('skip', params.skip.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const url = `/api/v1/admin/guest-users${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    return api.get<{
      total: number;
      items: Array<{
        id: number;
        email: string;
        is_active: boolean;
        max_conversations: number;
        conversations_used: number;
        expires_at: string;
        created_at: string;
        is_expired: boolean;
        is_limit_reached: boolean;
        can_access: boolean;
        days_remaining: number;
        conversations_remaining: number;
      }>;
    }>(url);
  },

  getGuestUser: async (id: number) => {
    return api.get<{
      id: number;
      email: string;
      is_active: boolean;
      max_conversations: number;
      conversations_used: number;
      expires_at: string;
      created_at: string;
      is_expired: boolean;
      is_limit_reached: boolean;
      can_access: boolean;
      days_remaining: number;
      conversations_remaining: number;
    }>(`/api/v1/admin/guest-users/${id}`);
  },

  updateGuestUser: async (
    id: number,
    data: {
      is_active?: boolean;
      max_conversations?: number;
      expires_at?: string;
    },
  ) => {
    return api.patch<{ message: string; guest_user: any }>(
      `/api/v1/admin/guest-users/${id}`,
      data,
    );
  },

  deleteGuestUser: async (id: number) => {
    return api.delete<{ message: string }>(
      `/api/v1/admin/guest-users/${id}`,
    );
  },
};

export default api;
