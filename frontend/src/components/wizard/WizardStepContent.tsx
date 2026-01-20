import { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { useWizard } from './WizardContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export interface WizardStepContentProps {
  step: number; // Changed from stepIndex to step for cleaner API
  children: ReactNode;
  className?: string;
  title?: string;
  description?: string;
  showCard?: boolean;
}

export function WizardStepContent({
  step: stepIndex,
  children,
  className,
  title,
  description,
  showCard = false, // Default to false - let children handle their own cards
}: WizardStepContentProps) {
  const { currentStep, steps } = useWizard();

  if (currentStep !== stepIndex) {
    return null;
  }

  const step = steps[stepIndex];
  const displayTitle = title || step?.title;
  const displayDescription = description || step?.description;

  if (!showCard) {
    return <div className={cn('animate-in fade-in-50 duration-300', className)}>{children}</div>;
  }

  return (
    <Card className={cn('animate-in fade-in-50 duration-300', className)}>
      {(displayTitle || displayDescription) && (
        <CardHeader>
          {displayTitle && <CardTitle>{displayTitle}</CardTitle>}
          {displayDescription && <CardDescription>{displayDescription}</CardDescription>}
        </CardHeader>
      )}
      <CardContent>{children}</CardContent>
    </Card>
  );
}
