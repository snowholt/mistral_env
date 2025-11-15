import { create } from 'zustand';
import { Call, HT813Status, CaptureStatus } from '../services/api';

interface AppState {
  // Calls
  calls: Call[];
  setCalls: (calls: Call[]) => void;
  addCall: (call: Call) => void;
  updateCall: (callId: string, updates: Partial<Call>) => void;
  removeCall: (callId: string) => void;

  // HT813 Device
  deviceStatus: HT813Status | null;
  setDeviceStatus: (status: HT813Status | null) => void;

  // Capture
  captureStatus: CaptureStatus | null;
  setCaptureStatus: (status: CaptureStatus | null) => void;

  // UI state
  selectedCallId: string | null;
  setSelectedCallId: (callId: string | null) => void;
}

export const useStore = create<AppState>((set) => ({
  // Calls
  calls: [],
  setCalls: (calls) => set({ calls }),
  addCall: (call) => set((state) => ({ calls: [...state.calls, call] })),
  updateCall: (callId, updates) =>
    set((state) => ({
      calls: state.calls.map((call) =>
        call.call_id === callId ? { ...call, ...updates } : call
      ),
    })),
  removeCall: (callId) =>
    set((state) => ({
      calls: state.calls.filter((call) => call.call_id !== callId),
    })),

  // HT813 Device
  deviceStatus: null,
  setDeviceStatus: (status) => set({ deviceStatus: status }),

  // Capture
  captureStatus: null,
  setCaptureStatus: (status) => set({ captureStatus: status }),

  // UI state
  selectedCallId: null,
  setSelectedCallId: (callId) => set({ selectedCallId: callId }),
}));
