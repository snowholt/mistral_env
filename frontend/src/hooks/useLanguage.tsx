import { ReactNode } from 'react';
import {
  LanguageProvider as BaseLanguageProvider,
  useLanguage as useBaseLanguage,
} from '@/contexts/LanguageContext';

type Language = 'en' | 'ar';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  toggleLanguage: () => void;
  isRTL: boolean;
  t: (key: string) => string;
  dir: 'ltr' | 'rtl';
}

export const LanguageProvider = ({ children }: { children: ReactNode }) => (
  <BaseLanguageProvider>{children}</BaseLanguageProvider>
);

export const useLanguage = (): LanguageContextType => {
  const { language, setLanguage, t, dir } = useBaseLanguage();
  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'ar' : 'en');
  };
  const isRTL = dir === 'rtl';

  return { language, setLanguage, toggleLanguage, isRTL, t, dir };
};