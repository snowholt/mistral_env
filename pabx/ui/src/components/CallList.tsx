import { useEffect } from 'react';
import { Phone, PhoneOff, PhoneIncoming } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { apiService, Call } from '../services/api';
import { useStore } from '../store/useStore';

export function CallList() {
  const { calls, setCalls, selectedCallId, setSelectedCallId } = useStore();

  useEffect(() => {
    const fetchCalls = async () => {
      try {
        const data = await apiService.getCalls();
        setCalls(data.calls);
      } catch (error) {
        console.error('Error fetching calls:', error);
      }
    };

    fetchCalls();
    const interval = setInterval(fetchCalls, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [setCalls]);

  const getStateIcon = (state: string) => {
    switch (state.toLowerCase()) {
      case 'ringing':
      case 'inviting':
        return <PhoneIncoming className="w-5 h-5" />;
      case 'active':
      case 'answered':
        return <Phone className="w-5 h-5" />;
      case 'ended':
      case 'cancelled':
        return <PhoneOff className="w-5 h-5" />;
      default:
        return <Phone className="w-5 h-5" />;
    }
  };

  const getStateBadgeClass = (state: string) => {
    switch (state.toLowerCase()) {
      case 'ringing':
      case 'inviting':
        return 'status-badge status-ringing';
      case 'active':
      case 'answered':
        return 'status-badge status-active';
      case 'ended':
      case 'cancelled':
        return 'status-badge status-ended';
      default:
        return 'status-badge';
    }
  };

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'N/A';
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch {
      return 'Invalid date';
    }
  };

  return (
    <div className="card">
      <h2>Active Calls ({calls.length})</h2>
      
      {calls.length === 0 ? (
        <p style={{ color: '#888', marginTop: '1rem' }}>No active calls</p>
      ) : (
        <div style={{ marginTop: '1rem' }}>
          {calls.map((call: Call) => (
            <div
              key={call.call_id}
              onClick={() => setSelectedCallId(call.call_id)}
              style={{
                padding: '1rem',
                marginBottom: '0.5rem',
                backgroundColor: selectedCallId === call.call_id ? '#2a2a2a' : '#1f1f1f',
                borderRadius: '4px',
                cursor: 'pointer',
                border: selectedCallId === call.call_id ? '2px solid #646cff' : '2px solid transparent',
                transition: 'all 0.2s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  {getStateIcon(call.state)}
                  <div>
                    <div style={{ fontWeight: '500' }}>
                      {call.from_user} → {call.to_user}
                    </div>
                    <div style={{ fontSize: '0.875rem', color: '#888' }}>
                      Started {formatTime(call.started_at)}
                    </div>
                  </div>
                </div>
                <span className={getStateBadgeClass(call.state)}>
                  {call.state}
                </span>
              </div>
              
              {call.remote_rtp_ip && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#888' }}>
                  RTP: {call.remote_rtp_ip}:{call.remote_rtp_port}
                  {call.recording_file && ' • Recording'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
