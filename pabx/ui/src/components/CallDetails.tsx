import { useEffect, useState } from 'react';
import { PhoneOff, Mic } from 'lucide-react';
import { apiService, CallStats } from '../services/api';
import { useStore } from '../store/useStore';

export function CallDetails() {
  const { selectedCallId, removeCall } = useStore();
  const [callStats, setCallStats] = useState<CallStats | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedCallId) {
      setCallStats(null);
      return;
    }

    const fetchStats = async () => {
      try {
        const stats = await apiService.getCallStats(selectedCallId);
        setCallStats(stats);
      } catch (error) {
        console.error('Error fetching call stats:', error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 2000);

    return () => clearInterval(interval);
  }, [selectedCallId]);

  const handleEndCall = async () => {
    if (!selectedCallId) return;

    setLoading(true);
    try {
      await apiService.endCall(selectedCallId);
      removeCall(selectedCallId);
    } catch (error) {
      console.error('Error ending call:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartRecording = async () => {
    if (!selectedCallId) return;

    setLoading(true);
    try {
      await apiService.startRecording(selectedCallId);
    } catch (error) {
      console.error('Error starting recording:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!selectedCallId) {
    return (
      <div className="card">
        <h2>Call Details</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>Select a call to view details</p>
      </div>
    );
  }

  if (!callStats) {
    return (
      <div className="card">
        <h2>Call Details</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>Loading...</p>
      </div>
    );
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="card">
      <h2>Call Details</h2>

      <div style={{ marginTop: '1rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h3>{callStats.from_user} → {callStats.to_user}</h3>
          <div style={{ color: '#888', marginTop: '0.5rem' }}>
            Call ID: {callStats.call_id}
          </div>
          <div style={{ marginTop: '0.5rem' }}>
            <span className={`status-badge status-${callStats.state.toLowerCase()}`}>
              {callStats.state}
            </span>
          </div>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <h3>Duration</h3>
          <div style={{ fontSize: '2rem', fontWeight: '500', marginTop: '0.5rem' }}>
            {formatDuration(callStats.duration)}
          </div>
        </div>

        {callStats.rtp_stats && (
          <div style={{ marginBottom: '1.5rem' }}>
            <h3>RTP Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Packets Sent</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {callStats.rtp_stats.packets_sent}
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Packets Received</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {callStats.rtp_stats.packets_received}
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Bytes Sent</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {(callStats.rtp_stats.bytes_sent / 1024).toFixed(1)} KB
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Bytes Received</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {(callStats.rtp_stats.bytes_received / 1024).toFixed(1)} KB
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Packet Loss</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {callStats.rtp_stats.packet_loss}
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Jitter</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                  {callStats.rtp_stats.jitter.toFixed(2)} ms
                </div>
              </div>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.5rem' }}>
          <button
            onClick={handleEndCall}
            disabled={loading || callStats.state === 'ENDED'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              backgroundColor: '#ef4444',
            }}
          >
            <PhoneOff className="w-4 h-4" />
            End Call
          </button>

          <button
            onClick={handleStartRecording}
            disabled={loading || callStats.state === 'ENDED'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <Mic className="w-4 h-4" />
            Record
          </button>
        </div>
      </div>
    </div>
  );
}
