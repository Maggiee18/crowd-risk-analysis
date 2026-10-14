// WebSocket hook for real-time connection management
import { useState, useEffect, useRef, useCallback } from 'react';
import { getToken } from '../services/api';

const WS_URL = (import.meta.env && import.meta.env.VITE_WS_URL) || 'ws://localhost:8000/ws';

export function useWebSocket(onMessage) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeout = useRef(null);
  const messageHandler = useRef(onMessage);

  useEffect(() => { messageHandler.current = onMessage; }, [onMessage]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // The server only accepts logged-in clients; wait until there is a token
    const token = getToken();
    if (!token) {
      reconnectTimeout.current = setTimeout(connect, 2000);
      return;
    }

    try {
      const ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);

      ws.onopen = () => {
        setIsConnected(true);
        console.log('[WS] Connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          if (messageHandler.current) messageHandler.current(data);
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('[WS] Disconnected — reconnecting in 3s');
        reconnectTimeout.current = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        ws.close();
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[WS] Connection failed:', err);
      reconnectTimeout.current = setTimeout(connect, 5000);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { isConnected, lastMessage, sendMessage, connect, disconnect };
}

export default useWebSocket;
