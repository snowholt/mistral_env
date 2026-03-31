/**
 * Inbox Page
 * 
 * WhatsApp conversations inbox with real-time WebSocket updates.
 * Shows conversation list and message thread.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  MessageSquare,
  Send,
  Search,
  User,
  Bot,
  Loader2,
  RefreshCw,
  Archive,
  Ban,
  CheckCheck,
  Clock,
  ArrowLeft,
  Phone,
} from 'lucide-react';
import { api, authApi, tokenManager } from '@/lib/api';
import { useAuth } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import { useToast } from '@/components/ui/use-toast';
import { cn } from '@/lib/utils';

const translations = {
  en: {
    title: 'Inbox',
    description: 'Manage your WhatsApp conversations',
    search: 'Search conversations...',
    noConversations: 'No conversations yet',
    noConversationsDescription: 'When customers message you on WhatsApp, they\'ll appear here.',
    typeMessage: 'Type a message...',
    send: 'Send',
    selectConversation: 'Select a conversation',
    selectConversationDescription: 'Choose a conversation from the list to view messages.',
    unread: 'unread',
    active: 'Active',
    archived: 'Archived',
    blocked: 'Blocked',
    archive: 'Archive',
    block: 'Block',
    markRead: 'Mark as read',
    today: 'Today',
    yesterday: 'Yesterday',
    aiResponse: 'AI Response',
    you: 'You',
    customer: 'Customer',
    connecting: 'Connecting...',
    connected: 'Connected',
    disconnected: 'Offline',
    loadMore: 'Load more',
    noBusinessSetup: 'No WhatsApp Business account connected',
    setupWhatsApp: 'Connect WhatsApp',
  },
  ar: {
    title: 'صندوق الوارد',
    description: 'إدارة محادثات واتساب الخاصة بك',
    search: 'البحث في المحادثات...',
    noConversations: 'لا توجد محادثات بعد',
    noConversationsDescription: 'عندما يراسلك العملاء على واتساب، سيظهرون هنا.',
    typeMessage: 'اكتب رسالة...',
    send: 'إرسال',
    selectConversation: 'اختر محادثة',
    selectConversationDescription: 'اختر محادثة من القائمة لعرض الرسائل.',
    unread: 'غير مقروءة',
    active: 'نشط',
    archived: 'مؤرشف',
    blocked: 'محظور',
    archive: 'أرشفة',
    block: 'حظر',
    markRead: 'تحديد كمقروء',
    today: 'اليوم',
    yesterday: 'أمس',
    aiResponse: 'رد الذكاء الاصطناعي',
    you: 'أنت',
    customer: 'العميل',
    connecting: 'جاري الاتصال...',
    connected: 'متصل',
    disconnected: 'غير متصل',
    loadMore: 'تحميل المزيد',
    noBusinessSetup: 'لا يوجد حساب واتساب للأعمال متصل',
    setupWhatsApp: 'ربط واتساب',
  },
};

interface Conversation {
  id: number;
  contact_phone: string;
  contact_name: string | null;
  status: 'active' | 'archived' | 'blocked';
  last_message_at: string;
  unread_count: number;
  last_message_preview: string | null;
  created_at: string;
}

interface Message {
  id: number;
  conversation_id: number;
  direction: 'inbound' | 'outbound';
  content: string;
  message_type: 'text' | 'image' | 'audio' | 'document';
  status: 'pending' | 'sent' | 'delivered' | 'read' | 'failed';
  is_ai_response: boolean;
  created_at: string;
  wa_message_id: string | null;
}

export default function Inbox() {
  const { user } = useAuth();
  const { language, isRTL } = useLanguage();
  const { toast } = useToast();
  const t = translations[language as keyof typeof translations] || translations.en;

  // State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [hasWhatsAppAccount, setHasWhatsAppAccount] = useState<boolean | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection
  const connectWebSocket = useCallback(async () => {
    const token = tokenManager.getAccessToken();
    if (!token) return;

    try {
      await authApi.getMe();
    } catch (error) {
      console.warn('[Inbox WS] Auth check failed, skipping connect:', error);
      return;
    }

    const freshToken = tokenManager.getAccessToken();
    if (!freshToken) return;

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v1/whatsapp/inbox/ws?token=${encodeURIComponent(freshToken)}`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('[Inbox WS] Connected');
        setWsConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (e) {
          console.error('[Inbox WS] Parse error:', e);
        }
      };

      wsRef.current.onclose = () => {
        console.log('[Inbox WS] Disconnected');
        setWsConnected(false);
        // Reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('[Inbox WS] Error:', error);
      };
    } catch (e) {
      console.error('[Inbox WS] Connection error:', e);
    }
  }, []);

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'new_message':
        // Add message to list if viewing that conversation
        if (selectedConversation?.id === data.conversation_id) {
          setMessages(prev => [...prev, data.message]);
          scrollToBottom();
        }
        // Update conversation list
        setConversations(prev => {
          const updated = prev.map(c => {
            if (c.id === data.conversation_id) {
              return {
                ...c,
                last_message_at: data.message.created_at,
                last_message_preview: data.message.content.slice(0, 100),
                unread_count: c.id === selectedConversation?.id ? 0 : c.unread_count + 1,
              };
            }
            return c;
          });
          // Sort by last message
          return updated.sort((a, b) => 
            new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime()
          );
        });
        break;

      case 'new_conversation':
        setConversations(prev => [data.conversation, ...prev]);
        break;

      case 'conversation_updated':
        setConversations(prev =>
          prev.map(c => c.id === data.conversation.id ? data.conversation : c)
        );
        break;
    }
  };

  // Fetch conversations
  const fetchConversations = async () => {
    setIsLoading(true);
    try {
      const response = await api.get<Conversation[]>('/api/v1/whatsapp/inbox/conversations');
      setConversations(response);
      setHasWhatsAppAccount(true);
    } catch (error: any) {
      if (error.status === 404) {
        setHasWhatsAppAccount(false);
      } else {
        console.error('Failed to fetch conversations:', error);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch messages for a conversation
  const fetchMessages = async (conversationId: number) => {
    setIsLoadingMessages(true);
    try {
      const response = await api.get<Message[]>(
        `/api/v1/whatsapp/inbox/conversations/${conversationId}/messages`
      );
      setMessages(response);
      scrollToBottom();

      // Subscribe to this conversation via WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'subscribe',
          conversation_id: conversationId,
        }));
      }
    } catch (error) {
      console.error('Failed to fetch messages:', error);
      toast({
        title: 'Error',
        description: 'Failed to load messages',
        variant: 'destructive',
      });
    } finally {
      setIsLoadingMessages(false);
    }
  };

  // Send message
  const handleSendMessage = async () => {
    if (!newMessage.trim() || !selectedConversation || isSending) return;

    setIsSending(true);
    const messageContent = newMessage.trim();
    setNewMessage('');

    // Optimistic update
    const tempMessage: Message = {
      id: Date.now(),
      conversation_id: selectedConversation.id,
      direction: 'outbound',
      content: messageContent,
      message_type: 'text',
      status: 'pending',
      is_ai_response: false,
      created_at: new Date().toISOString(),
      wa_message_id: null,
    };
    setMessages(prev => [...prev, tempMessage]);
    scrollToBottom();

    try {
      const response = await api.post<Message>(
        `/api/v1/whatsapp/inbox/conversations/${selectedConversation.id}/messages`,
        { content: messageContent }
      );
      // Replace temp message with real one
      setMessages(prev => prev.map(m => m.id === tempMessage.id ? response : m));
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove temp message on failure
      setMessages(prev => prev.filter(m => m.id !== tempMessage.id));
      setNewMessage(messageContent); // Restore message
      toast({
        title: 'Error',
        description: 'Failed to send message',
        variant: 'destructive',
      });
    } finally {
      setIsSending(false);
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const selectConversation = (conversation: Conversation) => {
    setSelectedConversation(conversation);
    fetchMessages(conversation.id);
  };

  // Format date/time
  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const isYesterday = date.toDateString() === yesterday.toDateString();

    if (isToday) {
      return date.toLocaleTimeString(language === 'ar' ? 'ar-SA' : 'en-US', {
        hour: '2-digit',
        minute: '2-digit',
      });
    } else if (isYesterday) {
      return t.yesterday;
    } else {
      return date.toLocaleDateString(language === 'ar' ? 'ar-SA' : 'en-US', {
        month: 'short',
        day: 'numeric',
      });
    }
  };

  // Get status icon
  const getStatusIcon = (status: Message['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-3 w-3 text-muted-foreground" />;
      case 'sent':
        return <CheckCheck className="h-3 w-3 text-muted-foreground" />;
      case 'delivered':
        return <CheckCheck className="h-3 w-3 text-blue-500" />;
      case 'read':
        return <CheckCheck className="h-3 w-3 text-green-500" />;
      default:
        return null;
    }
  };

  // Filter conversations by search
  const filteredConversations = conversations.filter(conv => {
    const search = searchQuery.toLowerCase();
    return (
      conv.contact_phone.includes(search) ||
      conv.contact_name?.toLowerCase().includes(search) ||
      conv.last_message_preview?.toLowerCase().includes(search)
    );
  });

  // Effects
  useEffect(() => {
    fetchConversations();
    void connectWebSocket();

    return () => {
      wsRef.current?.close();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  // No WhatsApp account state
  if (hasWhatsAppAccount === false) {
    return (
      <div className={`flex items-center justify-center min-h-[60vh] ${isRTL ? 'rtl' : 'ltr'}`}>
        <Card className="max-w-md w-full">
          <CardContent className="py-8 text-center">
            <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="font-medium text-lg">{t.noBusinessSetup}</h3>
            <p className="text-sm text-muted-foreground mt-2 mb-4">
              {t.noConversationsDescription}
            </p>
            <Button asChild>
              <a href="/app/whatsapp">{t.setupWhatsApp}</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={`h-[calc(100vh-8rem)] ${isRTL ? 'rtl' : 'ltr'}`}>
      <div className="flex h-full border rounded-lg overflow-hidden bg-card">
        {/* Conversation List */}
        <div className={cn(
          "w-full md:w-80 border-r flex flex-col",
          selectedConversation && "hidden md:flex"
        )}>
          {/* Header */}
          <div className="p-4 border-b space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold">{t.title}</h2>
              <Badge variant={wsConnected ? "default" : "secondary"} className="text-xs">
                {wsConnected ? t.connected : t.disconnected}
              </Badge>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder={t.search}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          {/* Conversation List */}
          <ScrollArea className="flex-1">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : filteredConversations.length === 0 ? (
              <div className="p-4 text-center">
                <MessageSquare className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground">{t.noConversations}</p>
              </div>
            ) : (
              <div className="divide-y">
                {filteredConversations.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => selectConversation(conv)}
                    className={cn(
                      "w-full p-3 text-left hover:bg-muted/50 transition-colors",
                      selectedConversation?.id === conv.id && "bg-muted"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <Avatar className="h-10 w-10">
                        <AvatarFallback>
                          {conv.contact_name?.charAt(0) || conv.contact_phone.slice(-2)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="font-medium truncate">
                            {conv.contact_name || conv.contact_phone}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {formatTime(conv.last_message_at)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between mt-1">
                          <p className="text-sm text-muted-foreground truncate">
                            {conv.last_message_preview || '...'}
                          </p>
                          {conv.unread_count > 0 && (
                            <Badge variant="default" className="h-5 min-w-5 text-xs">
                              {conv.unread_count}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Message Thread */}
        <div className={cn(
          "flex-1 flex flex-col",
          !selectedConversation && "hidden md:flex"
        )}>
          {selectedConversation ? (
            <>
              {/* Conversation Header */}
              <div className="p-4 border-b flex items-center gap-3">
                <Button
                  variant="ghost"
                  size="icon"
                  className="md:hidden"
                  onClick={() => setSelectedConversation(null)}
                >
                  <ArrowLeft className="h-5 w-5" />
                </Button>
                <Avatar>
                  <AvatarFallback>
                    {selectedConversation.contact_name?.charAt(0) ||
                      selectedConversation.contact_phone.slice(-2)}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <h3 className="font-medium">
                    {selectedConversation.contact_name || selectedConversation.contact_phone}
                  </h3>
                  <p className="text-sm text-muted-foreground flex items-center gap-1">
                    <Phone className="h-3 w-3" />
                    {selectedConversation.contact_phone}
                  </p>
                </div>
              </div>

              {/* Messages */}
              <ScrollArea className="flex-1 p-4">
                {isLoadingMessages ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2 className="h-6 w-6 animate-spin" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={cn(
                          "flex",
                          message.direction === 'outbound' ? 'justify-end' : 'justify-start'
                        )}
                      >
                        <div
                          className={cn(
                            "max-w-[70%] rounded-lg px-4 py-2",
                            message.direction === 'outbound'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          )}
                        >
                          {message.is_ai_response && (
                            <div className="flex items-center gap-1 text-xs opacity-70 mb-1">
                              <Bot className="h-3 w-3" />
                              {t.aiResponse}
                            </div>
                          )}
                          <p className="whitespace-pre-wrap break-words">{message.content}</p>
                          <div className={cn(
                            "flex items-center gap-1 text-xs mt-1",
                            message.direction === 'outbound' ? 'justify-end' : 'justify-start',
                            message.direction === 'outbound' ? 'opacity-70' : 'text-muted-foreground'
                          )}>
                            <span>
                              {new Date(message.created_at).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit',
                              })}
                            </span>
                            {message.direction === 'outbound' && getStatusIcon(message.status)}
                          </div>
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </ScrollArea>

              {/* Message Input */}
              <div className="p-4 border-t">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendMessage();
                  }}
                  className="flex gap-2"
                >
                  <Input
                    placeholder={t.typeMessage}
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    disabled={isSending}
                  />
                  <Button type="submit" disabled={!newMessage.trim() || isSending}>
                    {isSending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </form>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <MessageSquare className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="font-medium">{t.selectConversation}</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {t.selectConversationDescription}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
