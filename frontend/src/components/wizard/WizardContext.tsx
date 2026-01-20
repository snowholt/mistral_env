import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export interface WizardStep {
  id: string | number;
  title: string;
  description?: string;
  icon?: React.ElementType;
  isOptional?: boolean;
  isCompleted?: boolean;
}

export interface WizardContextType {
  steps: WizardStep[];
  currentStep: number;
  setCurrentStep: (step: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  goToStep: (stepIndex: number) => void;
  isFirstStep: boolean;
  isLastStep: boolean;
  canGoNext: boolean;
  setCanGoNext: (canGo: boolean) => void;
  completeStep: (stepIndex: number) => void;
  data: Record<string, any>;
  updateData: (newData: Record<string, any>) => void;
  setData: (data: Record<string, any>) => void;
  isSaving: boolean;
  setIsSaving: (saving: boolean) => void;
  resetWizard: () => void;
  totalSteps: number;
}

const WizardContext = createContext<WizardContextType | null>(null);

export interface WizardProviderProps {
  children: ReactNode;
  steps?: WizardStep[];
  totalSteps?: number;
  initialData?: Record<string, any>;
  onComplete?: (data: Record<string, any>) => void;
}

export function WizardProvider({ 
  children, 
  steps: initialSteps = [], 
  totalSteps: initialTotalSteps,
  initialData = {}, 
  onComplete 
}: WizardProviderProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [canGoNext, setCanGoNext] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [wizardSteps, setWizardSteps] = useState<WizardStep[]>(initialSteps);
  const [data, setData] = useState<Record<string, any>>(initialData);

  const totalSteps = initialTotalSteps || wizardSteps.length;
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  const nextStep = useCallback(() => {
    if (currentStep < totalSteps - 1 && canGoNext) {
      setCurrentStep(currentStep + 1);
    } else if (isLastStep && onComplete) {
      onComplete(data);
    }
  }, [currentStep, totalSteps, canGoNext, isLastStep, onComplete, data]);

  const prevStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const goToStep = useCallback((stepIndex: number) => {
    if (stepIndex >= 0 && stepIndex < totalSteps) {
      setCurrentStep(stepIndex);
    }
  }, [totalSteps]);

  const completeStep = useCallback((stepIndex: number) => {
    setWizardSteps(prev => prev.map((step, idx) => 
      idx === stepIndex ? { ...step, isCompleted: true } : step
    ));
  }, []);

  const updateData = useCallback((newData: Record<string, any>) => {
    setData(prev => ({ ...prev, ...newData }));
  }, []);

  const resetWizard = useCallback(() => {
    setCurrentStep(0);
    setData(initialData);
    setWizardSteps(initialSteps);
    setCanGoNext(true);
    setIsSaving(false);
  }, [initialData, initialSteps]);

  return (
    <WizardContext.Provider
      value={{
        steps: wizardSteps,
        currentStep,
        setCurrentStep,
        nextStep,
        prevStep,
        goToStep,
        isFirstStep,
        isLastStep,
        canGoNext,
        setCanGoNext,
        completeStep,
        data,
        updateData,
        setData,
        isSaving,
        setIsSaving,
        resetWizard,
        totalSteps,
      }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const context = useContext(WizardContext);
  if (!context) {
    throw new Error('useWizard must be used within a WizardProvider');
  }
  return context;
}
