import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { AgentConfig, AgentConfigRequest, AIControlRequest, Customer } from '@/types/agent';

interface CreateCustomerRequest {
  name: string;
  email: string;
}

// Fetch customers to get the ID
export function useCustomers() {
  return useQuery({
    queryKey: ['customers'],
    queryFn: async () => {
      return api.get<Customer[]>('/api/v1/whatsapp/customers');
    },
  });
}

// Create a new customer (business)
export function useCreateCustomer() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateCustomerRequest) => {
      return api.post<Customer>('/api/v1/whatsapp/customers', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['customers'] });
    },
  });
}

// Fetch agent configuration
export function useAgentConfig(customerId: number | undefined) {
  return useQuery({
    queryKey: ['agent-config', customerId],
    queryFn: async () => {
      if (!customerId) throw new Error('Customer ID is required');
      return api.get<AgentConfig>(`/api/v1/whatsapp/agents/config/${customerId}`);
    },
    enabled: !!customerId,
    retry: false, // Don't retry if 404 (not configured yet)
  });
}

// Update agent configuration
export function useUpdateAgentConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: AgentConfigRequest) => {
      return api.post<AgentConfig>('/api/v1/whatsapp/agents/configure', data);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['agent-config', data.customer_id] });
      queryClient.invalidateQueries({ queryKey: ['customers'] }); // Update "has_agent_config" flag
    },
  });
}

// Control AI (Pause/Resume)
export function useAIControl() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ customerId, data }: { customerId: number; data: AIControlRequest }) => {
      return api.post<{ success: boolean; message: string; ai_enabled: boolean; ai_pause_until: string | null }>(
        `/api/v1/whatsapp/agents/config/${customerId}/ai-control`,
        data
      );
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['agent-config', variables.customerId] });
    },
  });
}
