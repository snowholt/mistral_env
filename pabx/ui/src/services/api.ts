import axios from 'axios';

const API_BASE_URL = '/api';

export interface Call {
  call_id: string;
  from_user: string;
  to_user: string;
  state: string;
  started_at: string | null;
  answered_at: string | null;
  ended_at: string | null;
  local_rtp_port: number | null;
  remote_rtp_ip: string | null;
  remote_rtp_port: number | null;
  recording_file: string | null;
}

export interface CallStats {
  call_id: string;
  from_user: string;
  to_user: string;
  state: string;
  duration: number;
  rtp_stats: {
    packets_sent: number;
    packets_received: number;
    bytes_sent: number;
    bytes_received: number;
    packet_loss: number;
    jitter: number;
  } | null;
}

export interface HT813Status {
  mac_address: string;
  firmware_version: string;
  uptime: number;
  ip_address: string;
  fxs1_registered: boolean;
  fxs2_registered: boolean;
  active_calls: number;
}

export interface HT813Statistics {
  ports: Array<{
    port: string;
    total_calls: number;
    connected: number;
    failed: number;
    incoming: number;
    outgoing: number;
  }>;
}

export interface CaptureStatus {
  enabled: boolean;
  running?: boolean;
  statistics?: {
    running: boolean;
    duration_seconds: number;
    packets_captured: number;
    packets_stored: number;
    packets_dropped: number;
    capture_rate: number;
    packet_types: Record<string, number>;
  };
}

class ApiService {
  // Health check
  async healthCheck() {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  }

  // Call management
  async getCalls(): Promise<{ count: number; calls: Call[] }> {
    const response = await axios.get(`${API_BASE_URL}/calls`);
    return response.data;
  }

  async getCall(callId: string): Promise<Call> {
    const response = await axios.get(`${API_BASE_URL}/calls/${callId}`);
    return response.data;
  }

  async getCallStats(callId: string): Promise<CallStats> {
    const response = await axios.get(`${API_BASE_URL}/calls/${callId}/stats`);
    return response.data;
  }

  async answerCall(callId: string) {
    const response = await axios.post(`${API_BASE_URL}/calls/${callId}/answer`);
    return response.data;
  }

  async endCall(callId: string) {
    const response = await axios.post(`${API_BASE_URL}/calls/${callId}/end`);
    return response.data;
  }

  async playAudio(callId: string, audioFile: string) {
    const response = await axios.post(`${API_BASE_URL}/calls/${callId}/play`, null, {
      params: { audio_file: audioFile },
    });
    return response.data;
  }

  async startRecording(callId: string) {
    const response = await axios.post(`${API_BASE_URL}/calls/${callId}/record`);
    return response.data;
  }

  // HT813 device
  async getHT813Status(): Promise<HT813Status> {
    const response = await axios.get(`${API_BASE_URL}/ht813/status`);
    return response.data;
  }

  async getHT813Statistics(): Promise<HT813Statistics> {
    const response = await axios.get(`${API_BASE_URL}/ht813/statistics`);
    return response.data;
  }

  async rebootHT813() {
    const response = await axios.post(`${API_BASE_URL}/ht813/reboot`);
    return response.data;
  }

  // Packet capture
  async getCaptureStatus(): Promise<CaptureStatus> {
    const response = await axios.get(`${API_BASE_URL}/capture/status`);
    return response.data;
  }

  async getCaptureSessions() {
    const response = await axios.get(`${API_BASE_URL}/capture/sessions`);
    return response.data;
  }
}

export const apiService = new ApiService();
