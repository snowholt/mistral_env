import { useEffect, useState } from 'react';
import { Wifi, WifiOff, Activity, RefreshCw } from 'lucide-react';
import { apiService, HT813Statistics } from '../services/api';
import { useStore } from '../store/useStore';

export function DeviceDashboard() {
  const { deviceStatus, setDeviceStatus } = useStore();
  const [statistics, setStatistics] = useState<HT813Statistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDeviceInfo = async () => {
      try {
        setError(null);
        const [status, stats] = await Promise.all([
          apiService.getHT813Status(),
          apiService.getHT813Statistics(),
        ]);
        setDeviceStatus(status);
        setStatistics(stats);
      } catch (error) {
        console.error('Error fetching device info:', error);
        setError('Failed to connect to HT813 device');
      }
    };

    fetchDeviceInfo();
    const interval = setInterval(fetchDeviceInfo, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [setDeviceStatus]);

  const handleReboot = async () => {
    if (!confirm('Are you sure you want to reboot the HT813 device?')) {
      return;
    }

    setLoading(true);
    try {
      await apiService.rebootHT813();
      alert('Reboot command sent successfully');
    } catch (error) {
      console.error('Error rebooting device:', error);
      alert('Failed to reboot device');
    } finally {
      setLoading(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  if (error) {
    return (
      <div className="card">
        <h2>HT813 Device</h2>
        <div style={{ color: '#ef4444', marginTop: '1rem' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!deviceStatus) {
    return (
      <div className="card">
        <h2>HT813 Device</h2>
        <p style={{ color: '#888', marginTop: '1rem' }}>Loading device information...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>HT813 Device</h2>
        <button
          onClick={handleReboot}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.875rem',
          }}
        >
          <RefreshCw className="w-4 h-4" />
          Reboot
        </button>
      </div>

      <div style={{ marginTop: '1rem' }}>
        {/* Device Info */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>IP Address</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {deviceStatus.ip_address}
              </div>
            </div>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>MAC Address</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {deviceStatus.mac_address}
              </div>
            </div>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>Firmware</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {deviceStatus.firmware_version}
              </div>
            </div>
            <div>
              <div style={{ color: '#888', fontSize: '0.875rem' }}>Uptime</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '500', marginTop: '0.25rem' }}>
                {formatUptime(deviceStatus.uptime)}
              </div>
            </div>
          </div>
        </div>

        {/* FXS Port Status */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h3>FXS Ports</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
            <div
              style={{
                padding: '1rem',
                backgroundColor: '#1f1f1f',
                borderRadius: '4px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                {deviceStatus.fxs1_registered ? (
                  <Wifi className="w-5 h-5" style={{ color: '#10b981' }} />
                ) : (
                  <WifiOff className="w-5 h-5" style={{ color: '#ef4444' }} />
                )}
                <span style={{ fontWeight: '500' }}>FXS Port 1</span>
              </div>
              <span className={deviceStatus.fxs1_registered ? 'status-badge status-registered' : 'status-badge status-unregistered'}>
                {deviceStatus.fxs1_registered ? 'Registered' : 'Unregistered'}
              </span>
            </div>

            <div
              style={{
                padding: '1rem',
                backgroundColor: '#1f1f1f',
                borderRadius: '4px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                {deviceStatus.fxs2_registered ? (
                  <Wifi className="w-5 h-5" style={{ color: '#10b981' }} />
                ) : (
                  <WifiOff className="w-5 h-5" style={{ color: '#ef4444' }} />
                )}
                <span style={{ fontWeight: '500' }}>FXS Port 2</span>
              </div>
              <span className={deviceStatus.fxs2_registered ? 'status-badge status-registered' : 'status-badge status-unregistered'}>
                {deviceStatus.fxs2_registered ? 'Registered' : 'Unregistered'}
              </span>
            </div>
          </div>
        </div>

        {/* Call Statistics */}
        {statistics && (
          <div>
            <h3>Call Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
              {statistics.ports.map((port) => (
                <div
                  key={port.port}
                  style={{
                    padding: '1rem',
                    backgroundColor: '#1f1f1f',
                    borderRadius: '4px',
                  }}
                >
                  <div style={{ fontWeight: '500', marginBottom: '0.75rem' }}>
                    {port.port}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.875rem' }}>
                    <div>
                      <div style={{ color: '#888' }}>Total</div>
                      <div style={{ fontWeight: '500' }}>{port.total_calls}</div>
                    </div>
                    <div>
                      <div style={{ color: '#888' }}>Connected</div>
                      <div style={{ fontWeight: '500', color: '#10b981' }}>{port.connected}</div>
                    </div>
                    <div>
                      <div style={{ color: '#888' }}>Failed</div>
                      <div style={{ fontWeight: '500', color: '#ef4444' }}>{port.failed}</div>
                    </div>
                    <div>
                      <div style={{ color: '#888' }}>Incoming</div>
                      <div style={{ fontWeight: '500' }}>{port.incoming}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Active Calls Badge */}
        <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity className="w-5 h-5" />
          <span style={{ fontWeight: '500' }}>Active Calls:</span>
          <span style={{ fontSize: '1.25rem', fontWeight: '500', color: '#646cff' }}>
            {deviceStatus.active_calls}
          </span>
        </div>
      </div>
    </div>
  );
}
