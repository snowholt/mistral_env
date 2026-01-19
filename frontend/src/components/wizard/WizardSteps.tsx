import { cn } from '@/lib/utils';
import { useWizard, WizardStep } from './WizardContext';
import { Check } from 'lucide-react';

export interface WizardStepsProps {
  className?: string;
  orientation?: 'horizontal' | 'vertical';
  steps?: WizardStep[];
}

export function WizardSteps({ className, orientation = 'horizontal', steps: propSteps }: WizardStepsProps) {
  const { steps: contextSteps, currentStep, goToStep, totalSteps } = useWizard();
  
  // Use prop steps if provided, otherwise use context steps
  const steps = propSteps || contextSteps;
  const stepCount = steps.length || totalSteps;

  return (
    <nav
      className={cn(
        'flex',
        orientation === 'vertical' ? 'flex-col space-y-4' : 'items-center justify-between',
        className
      )}
      aria-label="Wizard steps"
    >
      {steps.map((step, index) => {
        const isActive = index === currentStep;
        const isCompleted = step.isCompleted || index < currentStep;
        const isClickable = isCompleted || index <= currentStep;
        const StepIcon = step.icon;

        return (
          <div
            key={step.id}
            className={cn(
              'flex items-center',
              orientation === 'horizontal' && index !== steps.length - 1 && 'flex-1'
            )}
          >
            <button
              type="button"
              onClick={() => isClickable && goToStep(index)}
              disabled={!isClickable}
              className={cn(
                'flex items-center gap-3 group',
                isClickable ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'
              )}
            >
              {/* Step indicator */}
              <div
                className={cn(
                  'flex h-10 w-10 items-center justify-center rounded-full border-2 transition-all',
                  isActive && 'border-primary bg-primary text-primary-foreground',
                  isCompleted && !isActive && 'border-primary bg-primary text-primary-foreground',
                  !isActive && !isCompleted && 'border-muted-foreground/30 bg-background text-muted-foreground'
                )}
              >
                {isCompleted && !isActive ? (
                  <Check className="h-5 w-5" />
                ) : StepIcon ? (
                  <StepIcon className="h-5 w-5" />
                ) : (
                  <span className="text-sm font-semibold">{index + 1}</span>
                )}
              </div>

              {/* Step label */}
              <div className="hidden sm:block">
                <p
                  className={cn(
                    'text-sm font-medium transition-colors',
                    isActive && 'text-primary',
                    isCompleted && !isActive && 'text-primary',
                    !isActive && !isCompleted && 'text-muted-foreground'
                  )}
                >
                  {step.title}
                </p>
                {step.description && (
                  <p className="text-xs text-muted-foreground">{step.description}</p>
                )}
              </div>
            </button>

            {/* Connector line */}
            {orientation === 'horizontal' && index !== steps.length - 1 && (
              <div
                className={cn(
                  'mx-4 h-0.5 flex-1 transition-colors',
                  isCompleted ? 'bg-primary' : 'bg-muted-foreground/30'
                )}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
