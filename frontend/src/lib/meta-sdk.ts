/**
 * Meta SDK Loader
 * 
 * Loads and initializes the Meta (Facebook) JavaScript SDK for
 * WhatsApp Embedded Signup flow.
 */

// Meta App ID from environment
const META_APP_ID = import.meta.env.VITE_META_APP_ID || '1438830807856481';

// SDK version
const SDK_VERSION = 'v21.0';

// Global FB SDK type declaration
declare global {
  interface Window {
    FB: {
      init: (params: { appId: string; autoLogAppEvents: boolean; xfbml: boolean; version: string }) => void;
      login: (
        callback: (response: { authResponse?: { accessToken: string; userID: string }; status: string }) => void,
        params: { config_id: string; response_type: string; override_default_response_type: boolean; extras: Record<string, any> }
      ) => void;
      getLoginStatus: (callback: (response: { status: string; authResponse?: { accessToken: string } }) => void) => void;
      logout: (callback?: () => void) => void;
      api: (path: string, callback: (response: any) => void) => void;
    };
    fbAsyncInit: () => void;
  }
}

// Track SDK loading state
let sdkLoadPromise: Promise<void> | null = null;
let isInitialized = false;

/**
 * Load the Meta SDK asynchronously.
 * Returns a promise that resolves when SDK is loaded and initialized.
 */
export async function loadMetaSDK(): Promise<void> {
  // Return existing promise if already loading
  if (sdkLoadPromise) {
    return sdkLoadPromise;
  }

  // Already loaded and initialized
  if (isInitialized && window.FB) {
    return Promise.resolve();
  }

  sdkLoadPromise = new Promise((resolve, reject) => {
    // Set up the init callback before loading
    window.fbAsyncInit = () => {
      try {
        window.FB.init({
          appId: META_APP_ID,
          autoLogAppEvents: true,
          xfbml: true,
          version: SDK_VERSION,
        });
        isInitialized = true;
        console.log('[MetaSDK] Initialized successfully');
        resolve();
      } catch (error) {
        console.error('[MetaSDK] Init error:', error);
        reject(error);
      }
    };

    // Check if script already exists
    if (document.getElementById('facebook-jssdk')) {
      // SDK script exists, try to init if FB is available
      if (window.FB) {
        window.fbAsyncInit();
      }
      return;
    }

    // Load the SDK script
    const script = document.createElement('script');
    script.id = 'facebook-jssdk';
    script.src = 'https://connect.facebook.net/en_US/sdk.js';
    script.async = true;
    script.defer = true;
    
    script.onerror = () => {
      console.error('[MetaSDK] Failed to load SDK script');
      reject(new Error('Failed to load Meta SDK'));
    };

    // Insert before first script
    const firstScript = document.getElementsByTagName('script')[0];
    firstScript?.parentNode?.insertBefore(script, firstScript);
  });

  return sdkLoadPromise;
}

/**
 * Start WhatsApp Embedded Signup flow.
 * Requires user to be logged in and OTP verified.
 */
export async function startWhatsAppSignup(
  configId: string,
  onSuccess: (code: string, waba_id: string, phone_number_id: string) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    await loadMetaSDK();
    
    if (!window.FB) {
      throw new Error('Meta SDK not available');
    }

    console.log('[MetaSDK] Starting WhatsApp Embedded Signup');
    
    window.FB.login(
      (response) => {
        if (response.authResponse) {
          const code = response.authResponse.accessToken;
          console.log('[MetaSDK] Login successful, received code');
          
          // The WABA ID and Phone Number ID will come from the extras
          // in a real implementation, they're returned via the callback URL
          // For now, we'll use placeholder values that the backend will exchange
          onSuccess(code, '', '');
        } else {
          console.error('[MetaSDK] Login failed or cancelled');
          onError('WhatsApp signup was cancelled or failed');
        }
      },
      {
        config_id: configId,
        response_type: 'code',
        override_default_response_type: true,
        extras: {
          sessionInfoVersion: 2, // Required for Embedded Signup
          setup: {
            // Pre-fill business category if known
            // business_verification: 'skip', // Skip for testing
          },
        },
      }
    );
  } catch (error) {
    console.error('[MetaSDK] Error starting signup:', error);
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
}

/**
 * Check if user is logged into Facebook.
 */
export async function checkFacebookLoginStatus(): Promise<{ loggedIn: boolean; userId?: string }> {
  try {
    await loadMetaSDK();
    
    return new Promise((resolve) => {
      window.FB.getLoginStatus((response) => {
        if (response.status === 'connected') {
          resolve({ loggedIn: true, userId: response.authResponse?.accessToken });
        } else {
          resolve({ loggedIn: false });
        }
      });
    });
  } catch {
    return { loggedIn: false };
  }
}

/**
 * Logout from Facebook.
 */
export async function facebookLogout(): Promise<void> {
  try {
    await loadMetaSDK();
    return new Promise((resolve) => {
      window.FB.logout(() => resolve());
    });
  } catch {
    // Ignore errors
  }
}
