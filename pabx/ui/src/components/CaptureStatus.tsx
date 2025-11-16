import { useEffect } from 'react';
import { Radio, CircleOff } from 'lucide-react';
import { apiService } from '../services/api';
import { useStore } from '../store/useStore';

export function CaptureStatus() {
  const { captureStatus, setCaptureStatus } = useStore();

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await apiService.getCaptureStatus();
        setCaptureStatus(status);
      } catch (error) {
        console.error('Error fetching capture status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);

    return () => clearInterval(interval);
  }, [setCaptureStatus]);

  if (!captureStatus) {
    return (
      <div className="card">
        <h2>Packet Capture</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>Loading...</p>
      </div>
    );
  }

  if (!captureStatus.enabled) {
    return (
      <div className="card">
        <h2>Packet Capture</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem', color: '#888' }}>
          <CircleOff className="w-5 h-5" />
          <span>Packet capture is disabled</span>
        </div>
        <p style={{ fontSize: '0.875rem', color: '#888', marginTop: '0.5rem' }}>
          Enable in config/settings.yaml to start capturing network traffic
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2>Packet Capture</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {captureStatus.running && (
            <>
              <Radio className="w-5 h-5" style={{ color: '#10b981' }} />
              <span className="status-badge status-active">Active</span>
            </>
          )}
        </div>
      </div>

      {captureStatus.statistics && (
        <div style={{ marginTop: '1rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>Total Packets</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {captureStatus.statistics.packets_captured?.toLocaleString() || 0}
              </div>
            </div>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>Capture Rate</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {(captureStatus.statistics.capture_rate || 0).toFixed(1)} pkt/s
              </div>
            </div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Stored</div>
                <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem' }}>
                  {captureStatus.statistics.packets_stored?.toLocaleString() || 0}
                </div>
              </div>
              <div>
                <div style={{ color: '#888', fontSize: '0.875rem' }}>Dropped</div>
                <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem', color: captureStatus.statistics.packets_dropped ? '#ef4444' : '#10b981' }}>
                  {captureStatus.statistics.packets_dropped || 0}
                </div>
              </div>
            </div>
          </div>

          {captureStatus.statistics.packet_types && Object.keys(captureStatus.statistics.packet_types).length > 0 && (
            <div>
              <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Packet Types</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                {Object.entries(captureStatus.statistics.packet_types).map(([type, count]) => (
                  <div
                    key={type}
                    style={{
                      padding: '0.75rem',
                      backgroundColor: '#1f1f1f',
                      borderRadius: '4px',
                    }}
                  >
                    <div style={{ fontSize: '0.875rem', color: '#888' }}>{type}</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: '500', marginTop: '0.25rem' }}>
                      {count}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
