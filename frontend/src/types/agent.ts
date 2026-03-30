export interface AgentConfig {
  id: number;
  customer_id: number;
  business_name: string;
  tone: 'professional' | 'friendly' | 'casual' | 'formal';
  behavior_rules?: string;
  custom_instructions?: string;
  system_prompt: string;
  ai_enabled: boolean;
  ai_pause_until?: string;
  ai_pause_duration_minutes: number;
  created_at: string;
  updated_at: string;
}

export interface AgentConfigRequest {
  customer_id: number;
  business_name: string;
  tone: 'professional' | 'friendly' | 'casual' | 'formal';
  behavior_rules?: string;
  custom_instructions?: string;
  ai_pause_duration_minutes: number;
}

export interface AIControlRequest {
  action: 'pause' | 'resume' | 'toggle';
  pause_minutes?: number;
}

export interface Customer {
  id: number;
  name: string;
  email: string;
  created_at: string;
  has_whatsapp: boolean;
  has_agent_config: boolean;
}
