import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type Language = 'en' | 'ar';

interface LanguageContextType {
  language: Language;
  toggleLanguage: () => void;
  isRTL: boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

// Define the key used in localStorage
const LANGUAGE_STORAGE_KEY = 'userLanguagePreference';

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  
  // 1. Initial State: Function to read from localStorage
  const [language, setLanguage] = useState<Language>(() => {
    // Check if we are running in the browser (client-side)
    if (typeof window !== 'undefined') {
      const storedLang = localStorage.getItem(LANGUAGE_STORAGE_KEY);
      // Return stored language, or default to 'ar' if nothing is found
      return (storedLang as Language) || 'ar'; 
    }
    // Return default for server-side rendering
    return 'ar'; 
  });

  // 2. Effect: Update localStorage whenever the language state changes
  useEffect(() => {
    // Check if we are running in the browser before accessing localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
    }
    // Also update the document's direction attribute for immediate effect if needed
    document.documentElement.setAttribute('lang', language);
  }, [language]); // Dependency array ensures this runs ONLY when 'language' changes

  const toggleLanguage = () => {
    // When toggling, set the new state, which will trigger the useEffect to update localStorage
    setLanguage(prev => prev === 'en' ? 'ar' : 'en');
  };

  const isRTL = language === 'ar';

  return (
    <LanguageContext.Provider value={{ language, toggleLanguage, isRTL }}>
      {/* Keeping the dir and className here is fine for immediate layout changes */}
      <div dir={isRTL ? 'rtl' : 'ltr'} className={isRTL ? 'font-arabic' : ''}>
        {children}
      </div>
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};