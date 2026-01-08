import { useEffect, useRef, useState } from 'react';
import { Call } from '../services/api';

export interface WebSocketEvent {
  type: 'call_incoming' | 'call_answered' | 'call_ended';
  data: Call;
}

export function useWebSocket(url: string = 'ws://localhost:8080/ws') {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WebSocketEvent | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<number>();

  const connect = () => {
    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket event:', data);
          setLastEvent(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000) as unknown as number;
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url]);

  return { isConnected, lastEvent };
}
