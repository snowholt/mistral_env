/**
 * Embeddable Chat Widget
 * 
 * A floating chat widget that customers can embed on their websites.
 * Features:
 * - Floating button to open/close
 * - Chat window with message history
 * - Real-time messaging via API
 * - Customizable colors via widget token
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageCircle, X, Send, Loader2, Minimize2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface WidgetConfig {
  widgetToken: string;
  apiUrl?: string;
  primaryColor?: string;
  headerText?: string;
  placeholderText?: string;
  welcomeMessage?: string;
  position?: 'bottom-right' | 'bottom-left';
}

const defaultConfig: Required<WidgetConfig> = {
  widgetToken: '',
  apiUrl: 'https://api.gmai.sa',
  primaryColor: '#0ea5e9',
  headerText: 'Chat with us',
  placeholderText: 'Type a message...',
  welcomeMessage: 'Hello! How can I help you today?',
  position: 'bottom-right',
};

export default function ChatWidget(props: WidgetConfig) {
  const config = { ...defaultConfig, ...props };
  
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionToken, setSessionToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize session and show welcome message
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      // Add welcome message
      setMessages([
        {
          id: 'welcome',
          role: 'assistant',
          content: config.welcomeMessage,
          timestamp: new Date(),
        },
      ]);
      
      // Create session
      createSession();
    }
  }, [isOpen]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  const createSession = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/api/v1/webchat/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          widget_token: config.widgetToken,
          visitor_id: getVisitorId(),
          page_url: window.location.href,
          referrer: document.referrer,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSessionToken(data.session_token);
      } else {
        console.error('Failed to create chat session');
      }
    } catch (err) {
      console.error('Error creating chat session:', err);
    }
  };

  const getVisitorId = (): string => {
    const key = 'gmai_visitor_id';
    let visitorId = localStorage.getItem(key);
    if (!visitorId) {
      visitorId = 'v_' + Math.random().toString(36).substring(2, 15);
      localStorage.setItem(key, visitorId);
    }
    return visitorId;
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading || !sessionToken) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${config.apiUrl}/api/v1/webchat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_token: sessionToken,
          message: userMessage.content,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (err) {
      setError('Failed to send message. Please try again.');
      console.error('Error sending message:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const positionClasses = config.position === 'bottom-left'
    ? 'left-4 sm:left-6'
    : 'right-4 sm:right-6';

  return (
    <>
      {/* Chat Window */}
      <div
        className={cn(
          'fixed bottom-20 z-[9999] w-[calc(100%-2rem)] sm:w-96 transition-all duration-300 transform',
          positionClasses,
          isOpen
            ? 'opacity-100 translate-y-0 pointer-events-auto'
            : 'opacity-0 translate-y-4 pointer-events-none'
        )}
        style={{ maxHeight: 'calc(100vh - 8rem)' }}
      >
        <div className="flex flex-col bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-200"
             style={{ height: '500px', maxHeight: 'calc(100vh - 8rem)' }}>
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3 text-white"
            style={{ backgroundColor: config.primaryColor }}
          >
            <span className="font-semibold">{config.headerText}</span>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 hover:bg-white/20 rounded-full transition-colors"
              aria-label="Close chat"
            >
              <Minimize2 className="h-5 w-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'flex',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                <div
                  className={cn(
                    'max-w-[80%] rounded-2xl px-4 py-2 text-sm',
                    message.role === 'user'
                      ? 'bg-blue-500 text-white rounded-br-sm'
                      : 'bg-white text-gray-800 border border-gray-200 rounded-bl-sm'
                  )}
                  style={
                    message.role === 'user'
                      ? { backgroundColor: config.primaryColor }
                      : undefined
                  }
                >
                  {message.content}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="text-center text-sm text-red-500">{error}</div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 bg-white border-t border-gray-200">
            <div className="flex items-center gap-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={config.placeholderText}
                disabled={isLoading || !sessionToken}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || !sessionToken}
                className="p-2 rounded-full text-white transition-all disabled:opacity-50 hover:scale-105"
                style={{ backgroundColor: config.primaryColor }}
                aria-label="Send message"
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'fixed bottom-4 sm:bottom-6 z-[9999] p-4 rounded-full text-white shadow-lg transition-all hover:scale-110',
          positionClasses
        )}
        style={{ backgroundColor: config.primaryColor }}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <MessageCircle className="h-6 w-6" />
        )}
      </button>
    </>
  );
}

/**
 * Standalone embed script for external websites.
 * 
 * Usage:
 * <script src="https://gmai.sa/widget.js" data-widget-token="xxx"></script>
 */
export function initChatWidget() {
  const script = document.currentScript;
  if (!script) return;

  const widgetToken = script.getAttribute('data-widget-token');
  if (!widgetToken) {
    console.error('GeniusAI Widget: Missing data-widget-token attribute');
    return;
  }

  const config: WidgetConfig = {
    widgetToken,
    apiUrl: script.getAttribute('data-api-url') || undefined,
    primaryColor: script.getAttribute('data-primary-color') || undefined,
    headerText: script.getAttribute('data-header-text') || undefined,
    placeholderText: script.getAttribute('data-placeholder-text') || undefined,
    welcomeMessage: script.getAttribute('data-welcome-message') || undefined,
    position: (script.getAttribute('data-position') as 'bottom-right' | 'bottom-left') || undefined,
  };

  // Create container
  const container = document.createElement('div');
  container.id = 'gmai-chat-widget';
  document.body.appendChild(container);

  // Mount React component (for standalone build)
  // This would need a separate build process to work as a standalone script
  console.log('GeniusAI Widget initialized with config:', config);
}
