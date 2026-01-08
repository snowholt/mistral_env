import { useEffect } from 'react';
import { Activity } from 'lucide-react';
import { CallList } from './components/CallList';
import { CallDetails } from './components/CallDetails';
import { DeviceDashboard } from './components/DeviceDashboard';
import { CaptureStatus } from './components/CaptureStatus';
import { useWebSocket } from './hooks/useWebSocket';
import { useStore } from './store/useStore';
import './index.css';

function App() {
  const { isConnected, lastEvent } = useWebSocket();
  const { addCall, updateCall, removeCall } = useStore();

  // Handle WebSocket events
  useEffect(() => {
    if (!lastEvent) return;

    switch (lastEvent.type) {
      case 'call_incoming':
        addCall(lastEvent.data);
        break;
      case 'call_answered':
        updateCall(lastEvent.data.call_id, lastEvent.data);
        break;
      case 'call_ended':
        removeCall(lastEvent.data.call_id);
        break;
    }
  }, [lastEvent, addCall, updateCall, removeCall]);

  return (
    <div>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Activity className="w-8 h-8" style={{ color: '#646cff' }} />
          <h1>BeautyAI PABX Dashboard</h1>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div 
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: isConnected ? '#10b981' : '#ef4444',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }}
          />
          <span style={{ fontSize: '0.875rem', color: '#888' }}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr',
        gap: '1rem',
        marginBottom: '1rem'
      }}>
        <CallList />
        <CallDetails />
      </div>

      {/* Device and Capture */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr',
        gap: '1rem'
      }}>
        <DeviceDashboard />
        <CaptureStatus />
      </div>

      {/* Footer */}
      <div style={{ 
        marginTop: '2rem',
        padding: '1rem',
        textAlign: 'center',
        color: '#888',
        fontSize: '0.875rem'
      }}>
        BeautyAI PABX v2.0 • Built with React + TypeScript + Vite
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
