import React, { useState, useEffect, useCallback } from 'react';
import { Calendar, Clock, User, Phone, Mail, CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';

const API_BASE = import.meta.env.VITE_API_URL || 'https://api.gmai.sa';

interface Customer {
  id: number;
  first_name: string;
  last_name: string;
  full_name: string;
  phone?: string;
  email?: string;
  preferred_language: string;
  created_at?: string;
}

interface TimeSlot {
  id: number;
  date: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  is_available: boolean;
  slots_remaining: number;
}

interface Appointment {
  id: number;
  customer: Customer;
  time_slot: TimeSlot;
  service_type: string;
  status: 'pending' | 'confirmed' | 'cancelled' | 'completed' | 'no_show';
  notes?: string;
  created_at: string;
  confirmed_at?: string;
}

interface ToolCallEvent {
  type: 'tool_call';
  tool: string;
  status: 'executing' | 'complete' | 'error';
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  allows_interruption?: boolean;
  error?: string;
}

interface AppointmentPanelProps {
  language: 'ar' | 'en';
  onToolCall?: (event: ToolCallEvent) => void;
}

const translations = {
  en: {
    customerInfo: 'Customer Info',
    appointments: 'Appointments',
    availableSlots: 'Available Slots',
    toolActivity: 'Tool Activity',
    noCustomer: 'No customer selected',
    noAppointments: 'No appointments',
    noSlots: 'No available slots',
    refresh: 'Refresh',
    loading: 'Loading...',
    status: 'Status',
    date: 'Date',
    time: 'Time',
    service: 'Service',
    name: 'Name',
    phone: 'Phone',
    email: 'Email',
    language: 'Language',
    executing: 'Executing',
    completed: 'Completed',
    error: 'Error',
    generateSlots: 'Generate Demo Slots',
    generatingSlots: 'Generating...',
    slotsGenerated: 'Slots generated successfully',
  },
  ar: {
    customerInfo: 'معلومات العميل',
    appointments: 'المواعيد',
    availableSlots: 'الأوقات المتاحة',
    toolActivity: 'نشاط الأداة',
    noCustomer: 'لم يتم تحديد عميل',
    noAppointments: 'لا توجد مواعيد',
    noSlots: 'لا توجد أوقات متاحة',
    refresh: 'تحديث',
    loading: 'جاري التحميل...',
    status: 'الحالة',
    date: 'التاريخ',
    time: 'الوقت',
    service: 'الخدمة',
    name: 'الاسم',
    phone: 'الهاتف',
    email: 'البريد',
    language: 'اللغة',
    executing: 'قيد التنفيذ',
    completed: 'مكتمل',
    error: 'خطأ',
    generateSlots: 'إنشاء مواعيد تجريبية',
    generatingSlots: 'جاري الإنشاء...',
    slotsGenerated: 'تم إنشاء المواعيد بنجاح',
  },
};

const statusColors: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-950/40 dark:text-yellow-300',
  confirmed: 'bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-300',
  cancelled: 'bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300',
  completed: 'bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300',
  no_show: 'bg-gray-100 text-gray-800 dark:bg-gray-900/60 dark:text-gray-200',
};

export function AppointmentPanel({ language, onToolCall }: AppointmentPanelProps) {
  const t = translations[language];
  const isRTL = language === 'ar';

  // State
  const [customer, setCustomer] = useState<Customer | null>(null);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [availableSlots, setAvailableSlots] = useState<TimeSlot[]>([]);
  const [toolActivity, setToolActivity] = useState<ToolCallEvent[]>([]);
  const [isLoadingSlots, setIsLoadingSlots] = useState(false);
  const [isGeneratingSlots, setIsGeneratingSlots] = useState(false);

  // Fetch available slots
  const fetchSlots = useCallback(async () => {
    setIsLoadingSlots(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/appointments/slots?days_ahead=7`);
      if (response.ok) {
        const data = await response.json();
        setAvailableSlots(data.available_slots || []);
      }
    } catch (error) {
      console.error('Failed to fetch slots:', error);
    } finally {
      setIsLoadingSlots(false);
    }
  }, []);

  // Fetch customer appointments
  const fetchAppointments = useCallback(async (customerId: number) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/appointments/customer/${customerId}/appointments`);
      if (response.ok) {
        const data = await response.json();
        setAppointments(data.appointments || []);
      }
    } catch (error) {
      console.error('Failed to fetch appointments:', error);
    }
  }, []);

  // Generate demo slots
  const generateSlots = async () => {
    setIsGeneratingSlots(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/appointments/slots/generate?days=7`, {
        method: 'POST',
      });
      if (response.ok) {
        await fetchSlots();
      }
    } catch (error) {
      console.error('Failed to generate slots:', error);
    } finally {
      setIsGeneratingSlots(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchSlots();
  }, [fetchSlots]);

  // Handle tool call events from parent (VoiceDemo)
  const handleToolCallEvent = useCallback((event: ToolCallEvent) => {
    // Add to activity log
    setToolActivity(prev => [event, ...prev].slice(0, 10));

    // Update state based on tool result
    if (event.status === 'complete' && event.result) {
      const result = event.result as Record<string, unknown>;
      
      if (event.tool === 'check_customer' || event.tool === 'register_customer') {
        if (result.customer) {
          const customerData = result.customer as Customer;
          setCustomer(customerData);
          if (customerData.id) {
            fetchAppointments(customerData.id);
          }
        }
      }
      
      if (event.tool === 'book_appointment' || event.tool === 'cancel_appointment') {
        // Refresh slots and appointments
        fetchSlots();
        if (customer?.id) {
          fetchAppointments(customer.id);
        }
      }
      
      if (event.tool === 'list_available_slots') {
        if (result.available_slots) {
          setAvailableSlots(result.available_slots as TimeSlot[]);
        }
      }
    }
    
    // Notify parent
    onToolCall?.(event);
  }, [customer, fetchSlots, fetchAppointments, onToolCall]);

  // Expose handler for parent component
  useEffect(() => {
    // Store the handler on the window for parent access
    (window as any).__appointmentPanelHandler = handleToolCallEvent;
    return () => {
      delete (window as any).__appointmentPanelHandler;
    };
  }, [handleToolCallEvent]);

  return (
    <div className={`space-y-4 ${isRTL ? 'text-right' : 'text-left'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Customer Info Card */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <User className="h-4 w-4" />
            {t.customerInfo}
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          {customer ? (
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <User className="h-3 w-3 text-muted-foreground" />
                <span className="font-medium">{customer.full_name}</span>
              </div>
              {customer.phone && (
                <div className="flex items-center gap-2">
                  <Phone className="h-3 w-3 text-muted-foreground" />
                  <span>{customer.phone}</span>
                </div>
              )}
              {customer.email && (
                <div className="flex items-center gap-2">
                  <Mail className="h-3 w-3 text-muted-foreground" />
                  <span>{customer.email}</span>
                </div>
              )}
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">
                  {customer.preferred_language === 'ar' ? 'العربية' : 'English'}
                </Badge>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground italic">{t.noCustomer}</p>
          )}
        </CardContent>
      </Card>

      {/* Appointments Card */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Calendar className="h-4 w-4" />
            {t.appointments}
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          {appointments.length > 0 ? (
            <ScrollArea className="h-[150px]">
              <div className="space-y-2">
                {appointments.map((appt) => (
                  <div
                    key={appt.id}
                    className="p-2 rounded border bg-muted/50 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{appt.time_slot.date}</span>
                      <Badge className={statusColors[appt.status] || 'bg-muted text-muted-foreground'}>
                        {appt.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>{appt.time_slot.start_time} - {appt.time_slot.end_time}</span>
                    </div>
                    <div className="text-muted-foreground">{appt.service_type}</div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <p className="text-sm text-muted-foreground italic">{t.noAppointments}</p>
          )}
        </CardContent>
      </Card>

      {/* Available Slots Card */}
      <Card>
        <CardHeader className="py-3 flex flex-row items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Clock className="h-4 w-4" />
            {t.availableSlots}
          </CardTitle>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchSlots}
              disabled={isLoadingSlots}
              className="h-6 w-6 p-0"
            >
              <RefreshCw className={`h-3 w-3 ${isLoadingSlots ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="py-2">
          {isLoadingSlots ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {t.loading}
            </div>
          ) : availableSlots.length > 0 ? (
            <ScrollArea className="h-[120px]">
              <div className="grid grid-cols-2 gap-1">
                {availableSlots.slice(0, 12).map((slot) => (
                  <div
                    key={slot.id}
                    className="p-1.5 rounded border text-xs bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900/60"
                  >
                    <div className="font-medium">{slot.date}</div>
                    <div className="text-muted-foreground">{slot.start_time}</div>
                  </div>
                ))}
              </div>
              {availableSlots.length > 12 && (
                <p className="text-xs text-muted-foreground mt-2">
                  +{availableSlots.length - 12} more slots
                </p>
              )}
            </ScrollArea>
          ) : (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground italic">{t.noSlots}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={generateSlots}
                disabled={isGeneratingSlots}
                className="w-full text-xs"
              >
                {isGeneratingSlots ? (
                  <>
                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                    {t.generatingSlots}
                  </>
                ) : (
                  t.generateSlots
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Tool Activity Card */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm flex items-center gap-2">
            ⚡ {t.toolActivity}
          </CardTitle>
        </CardHeader>
        <CardContent className="py-2">
          {toolActivity.length > 0 ? (
            <ScrollArea className="h-[100px]">
              <div className="space-y-1">
                {toolActivity.map((event, idx) => (
                  <div
                    key={idx}
                    className={`p-1.5 rounded text-xs flex items-center gap-2 ${
                      event.status === 'executing' ? 'bg-blue-50 dark:bg-blue-950/40' :
                      event.status === 'complete' ? 'bg-green-50 dark:bg-green-950/40' :
                      'bg-red-50 dark:bg-red-950/40'
                    }`}
                  >
                    {event.status === 'executing' ? (
                      <Loader2 className="h-3 w-3 animate-spin text-blue-600 dark:text-blue-400" />
                    ) : event.status === 'complete' ? (
                      <CheckCircle className="h-3 w-3 text-green-600 dark:text-green-400" />
                    ) : (
                      <XCircle className="h-3 w-3 text-red-600 dark:text-red-400" />
                    )}
                    <span className="font-mono">{event.tool}</span>
                    <Badge variant="outline" className="text-[10px] ml-auto">
                      {event.status === 'executing' ? t.executing :
                       event.status === 'complete' ? t.completed : t.error}
                    </Badge>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <p className="text-xs text-muted-foreground italic">No tool activity yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default AppointmentPanel;
