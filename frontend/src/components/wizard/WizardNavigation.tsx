import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useWizard } from './WizardContext';
import { ChevronLeft, ChevronRight, Loader2, Save } from 'lucide-react';

export interface WizardNavigationProps {
  className?: string;
  onSave?: () => void;
  onComplete?: () => void;
  isSaving?: boolean;
  showSaveButton?: boolean;
  nextLabel?: string;
  prevLabel?: string;
  completeLabel?: string;
  canProceed?: boolean;
  totalSteps?: number; // Optional, uses context if not provided
}

export function WizardNavigation({
  className,
  onSave,
  onComplete,
  isSaving: propIsSaving,
  showSaveButton = true,
  nextLabel = 'Next',
  prevLabel = 'Back',
  completeLabel = 'Complete Setup',
  canProceed = true,
}: WizardNavigationProps) {
  const { nextStep, prevStep, isFirstStep, isLastStep, canGoNext, isSaving: contextIsSaving } = useWizard();

  const isSaving = propIsSaving ?? contextIsSaving;

  const handleNext = () => {
    if (isLastStep && onComplete) {
      onComplete();
    } else if (isLastStep && onSave) {
      onSave();
    } else {
      nextStep();
    }
  };

  return (
    <div className={cn('flex items-center justify-between pt-6 border-t', className)}>
      <div>
        {!isFirstStep && (
          <Button
            type="button"
            variant="outline"
            onClick={prevStep}
            disabled={isSaving}
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            {prevLabel}
          </Button>
        )}
      </div>

      <div className="flex items-center gap-3">
        {showSaveButton && onSave && !isLastStep && (
          <Button
            type="button"
            variant="outline"
            onClick={onSave}
            disabled={isSaving}
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save Progress
          </Button>
        )}

        <Button
          type="button"
          onClick={handleNext}
          disabled={!canGoNext || !canProceed || isSaving}
        >
          {isSaving && isLastStep && (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          )}
          {isLastStep ? completeLabel : nextLabel}
          {!isLastStep && <ChevronRight className="h-4 w-4 ml-2" />}
        </Button>
      </div>
    </div>
  );
}
