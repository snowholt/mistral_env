import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, MicOff, Phone, PhoneOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAuth } from '@/hooks/useAuth';
import { useLanguage } from '@/hooks/useLanguage';
import { guestApi } from '@/lib/api';

// Helper to check if user has guest-level access
// Supports both guest-login flow (isGuest + guestUser) and unified auth (user.role === 'guest')
const hasGuestAccess = (isGuest: boolean, guestUser: any, user: any, isAdmin: boolean): boolean => {
  // Admin can always access
  if (isAdmin) return true;
  // Guest login flow
  if (isGuest && guestUser) return true;
  // Unified auth: regular login with guest role
  if (user?.role === 'guest') return true;
  return false;
};

interface Message {
  role: 'user' | 'assistant' | 'system';
  text: string;
  isRTL?: boolean;
}

interface Metrics {
  tps?: number;
  llm_latency?: number;
  stt_time?: number;
  tts_time?: number;
}

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'listening' | 'processing' | 'speaking';

const STUN_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' }
];

const API_BASE = import.meta.env.VITE_API_URL || 'https://api.gmai.sa';

const translations = {
  en: {
    error: 'Error',
    backToDashboard: 'Back to Dashboard',
    connecting: 'Connecting...',
  },
  ar: {
    error: 'خطأ',
    backToDashboard: 'العودة إلى لوحة التحكم',
    connecting: 'جاري الاتصال...',
  },
};

export default function VoiceDemo() {
  const { language: appLanguage } = useLanguage();
  const navigate = useNavigate();
  const { guestUser, isGuest, user, isAdmin, isAuthenticated } = useAuth();

  const t = translations[appLanguage as keyof typeof translations] || translations.en;

  // WebRTC refs
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const iceCandidateQueueRef = useRef<RTCIceCandidate[]>([]);
  const remoteDescriptionSetRef = useRef(false);
  const sessionIdRef = useRef<string | null>(null);
  const pendingLocalIceCandidatesRef = useRef<RTCIceCandidate[]>([]);
  const isTTSPlayingRef = useRef<boolean>(false);  // Track TTS playback state
  const pendingMicEnableRef = useRef<boolean>(false);  // Queue mic enable until TTS finishes
  const currentAssistantMessageRef = useRef<string>('');  // Ref to avoid stale closure

  // State
  const [language, setLanguage] = useState<'ar' | 'en'>(appLanguage === 'ar' ? 'ar' : 'en');
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState<string>('');
  const [metrics, setMetrics] = useState<Metrics>({});
  const [vadStatus, setVadStatus] = useState<string>('🔇 Mic Muted');
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // Refs for chat scroll
  const chatBoxRef = useRef<HTMLDivElement>(null);

  // Check guest access on mount
  useEffect(() => {
    // Check if user has guest-level access (either via guest login or unified auth with guest role)
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }
    
    // Allow access for: guest login users, unified auth guest users, or admins
    const canAccess = hasGuestAccess(isGuest, guestUser, user, isAdmin);
    if (!canAccess) {
      // Regular users without guest access get redirected to main app
      navigate('/app');
      return;
    }

    // Validate access for guest users (check limits, expiration)
    // Only do validation if we have actual guest user data (guest login flow)
    if (isGuest && guestUser) {
      const validateAccess = async () => {
        try {
          const validation = await guestApi.validateAccess();
          if (!validation.can_access) {
            if (validation.is_expired) {
              setError('Your demo access has expired. Please contact support.');
            } else if (validation.is_limit_reached) {
              setError('You have reached the maximum number of conversations for your demo.');
            } else {
              setError('Access denied. Please contact support.');
            }
          }
        } catch (err: any) {
          console.error('Access validation failed:', err);
          setError(err.response?.data?.message || 'Failed to validate access');
        }
      };
      validateAccess();
    }
    // For unified auth guest users (user.role === 'guest'), validation happens server-side
  }, [isGuest, guestUser, user, isAdmin, isAuthenticated, navigate]);

  // Auto-scroll chat
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, currentAssistantMessage]);

  // Detect Arabic text
  const isArabicText = (text: string): boolean => {
    const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/;
    return arabicPattern.test(text);
  };

  // Add message to chat
  const addMessage = (role: Message['role'], text: string) => {
    const isRTL = isArabicText(text);
    setMessages(prev => [...prev, { role, text, isRTL }]);
  };

  // Update assistant message (streaming)
  const updateAssistantMessage = (chunk: string) => {
    currentAssistantMessageRef.current += chunk;
    setCurrentAssistantMessage(currentAssistantMessageRef.current);
  };

  // Commit assistant message - uses ref to avoid stale closure issues
  const commitAssistantMessage = () => {
    const msgText = currentAssistantMessageRef.current;
    if (msgText) {
      console.log('💬 Committing assistant message:', msgText.substring(0, 50) + '...');
      const isRTL = isArabicText(msgText);
      setMessages(prev => [...prev, { role: 'assistant', text: msgText, isRTL }]);
      currentAssistantMessageRef.current = '';
      setCurrentAssistantMessage('');
    }
  };

  // Handle microphone control
  const handleMicControl = (enable: boolean) => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = enable;
      });
      setIsMicMuted(!enable);
      console.log(`🎤 Microphone ${enable ? 'enabled' : 'disabled'}`);
    }
  };

  // Play TTS audio
  const playTTSAudio = (base64Audio: string) => {
    try {
      // Stop any currently playing audio
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
        audioPlayerRef.current = null;
      }

      // Mark TTS as playing
      isTTSPlayingRef.current = true;

      // Decode base64 to binary
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Create audio blob and play
      const blob = new Blob([bytes], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(blob);
      const audioPlayer = new Audio(audioUrl);
      audioPlayerRef.current = audioPlayer;

      audioPlayer.onended = () => {
        console.log('🔊 TTS playback finished');
        URL.revokeObjectURL(audioUrl);
        audioPlayerRef.current = null;
        isTTSPlayingRef.current = false;
        
        // Enable mic if it was queued during TTS playback
        if (pendingMicEnableRef.current) {
          console.log('🎤 Enabling mic after TTS finished');
          handleMicControl(true);
          pendingMicEnableRef.current = false;
        }
      };

      audioPlayer.onerror = (e) => {
        console.error('❌ TTS playback error:', e);
        URL.revokeObjectURL(audioUrl);
        audioPlayerRef.current = null;
        isTTSPlayingRef.current = false;
        
        // Enable mic if it was queued
        if (pendingMicEnableRef.current) {
          handleMicControl(true);
          pendingMicEnableRef.current = false;
        }
      };

      audioPlayer.play().then(() => {
        console.log('🔊 TTS playback started');
      }).catch(e => {
        console.error('❌ TTS play failed:', e);
        isTTSPlayingRef.current = false;
        addMessage('system', '⚠️ Click anywhere to enable audio playback');
        
        // Enable mic if it was queued
        if (pendingMicEnableRef.current) {
          handleMicControl(true);
          pendingMicEnableRef.current = false;
        }
      });
    } catch (e) {
      console.error('❌ TTS audio error:', e);
      isTTSPlayingRef.current = false;
    }
  };

  // Handle state changes
  const handleStateChange = (state: string) => {
    if (state === 'processing') {
      setConnectionState('processing');
      setVadStatus('🔇 Mic Muted');
    } else if (state === 'speaking') {
      setConnectionState('speaking');
      setVadStatus('🔇 Mic Muted (TTS)');
    } else if (state === 'listening') {
      setConnectionState('listening');
      setVadStatus('🎤 Mic Active');
      commitAssistantMessage();
    }
  };

  // Setup data channel
  const setupDataChannel = (dc: RTCDataChannel) => {
    dcRef.current = dc;

    dc.onopen = () => {
      console.log('✅ Data channel opened');
      setConnectionState('connected');
      addMessage('system', language === 'ar' ? 'متصل! ابدأ في الحديث...' : 'Connected! Start speaking...');
    };

    dc.onclose = () => {
      console.log('❌ Data channel closed');
      setConnectionState('disconnected');
      addMessage('system', 'Session ended.');
    };

    dc.onerror = (error) => {
      console.error('❌ Data channel error:', error);
    };

    dc.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📨 Received:', data.type);

        switch (data.type) {
          case 'transcription':
            if (data.text) {
              // Commit any pending assistant message before starting new turn
              commitAssistantMessage();
              addMessage('user', data.text);
            }
            break;

          case 'response_chunk':
            if (data.text) {
              updateAssistantMessage(data.text);
            }
            break;

          case 'state':
            handleStateChange(data.state);
            break;

          case 'metrics':
            setMetrics({
              tps: data.tps,
              llm_latency: data.llm_latency,
              stt_time: data.stt_time,
              tts_time: data.tts_time
            });
            break;

          case 'mic_control':
            // Handle both formats: {enable: bool} and {action: "mute"/"unmute"}
            let shouldEnable: boolean;
            if (data.enable !== undefined) {
              shouldEnable = data.enable;
            } else if (data.action !== undefined) {
              shouldEnable = data.action === 'unmute';
            } else {
              console.warn('⚠️ mic_control missing enable/action field:', data);
              break;
            }
            
            // If trying to enable mic but TTS is still playing, queue it
            if (shouldEnable && isTTSPlayingRef.current) {
              console.log('🎤 Mic enable queued (TTS still playing)');
              pendingMicEnableRef.current = true;
            } else {
              handleMicControl(shouldEnable);
              if (!shouldEnable) {
                pendingMicEnableRef.current = false;  // Clear pending if muting
              }
            }
            break;

          case 'tts_audio':
            // Backend sends audio_base64, handle both field names for compatibility
            const audioData = data.audio_base64 || data.audio_data;
            if (audioData) {
              console.log('🔊 Received TTS audio, length:', audioData.length);
              playTTSAudio(audioData);
            } else {
              console.warn('⚠️ tts_audio received but no audio data found in payload:', Object.keys(data));
            }
            break;

          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (e) {
        console.error('❌ Error parsing message:', e);
      }
    };
  };

  // Process ICE candidate queue
  const processIceCandidateQueue = async () => {
    const pc = pcRef.current;
    if (!pc || !remoteDescriptionSetRef.current) return;

    console.log(`🧊 Processing ${iceCandidateQueueRef.current.length} queued ICE candidates`);

    for (const candidate of iceCandidateQueueRef.current) {
      try {
        await pc.addIceCandidate(candidate);
        console.log('✅ Added queued ICE candidate:', candidate.candidate?.substring(0, 50));
      } catch (e) {
        console.error('❌ Error adding queued ICE candidate:', e);
      }
    }

    iceCandidateQueueRef.current = [];
  };

  // Start session
  const startSession = async () => {
    try {
      setIsLoading(true);
      setError('');
      setMessages([]);
      setCurrentAssistantMessage('');
      setMetrics({});

      // Request microphone access
      console.log('🎤 Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000
        }
      });
      localStreamRef.current = stream;
      console.log('✅ Microphone access granted');

      // Create peer connection
      const pc = new RTCPeerConnection({ iceServers: STUN_SERVERS });
      pcRef.current = pc;

      // Add audio track
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
        console.log('✅ Added audio track:', track.kind);
      });

      // Setup data channel
      const dc = pc.createDataChannel('events', { ordered: true });
      setupDataChannel(dc);

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          console.log('🧊 New ICE candidate');
          
          // Queue if we don't have session_id yet
          if (!sessionIdRef.current) {
            console.log('🧊 Queuing ICE candidate (no session_id yet)');
            pendingLocalIceCandidatesRef.current.push(event.candidate);
            return;
          }
          
          // Send with session_id
          fetch(`${API_BASE}/api/v1/webrtc/voice/ice`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('beautyai_access_token')}`
            },
            body: JSON.stringify({
              session_id: sessionIdRef.current,
              candidate: event.candidate.candidate,
              sdp_mid: event.candidate.sdpMid,
              sdp_m_line_index: event.candidate.sdpMLineIndex
            })
          }).catch(e => console.error('❌ Error sending ICE candidate:', e));
        }
      };

      // Monitor connection state
      pc.onconnectionstatechange = () => {
        console.log('🔗 Connection state:', pc.connectionState);
        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          stopSession();
        }
      };

      // Create and send offer
      console.log('📤 Creating offer...');
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const response = await fetch(`${API_BASE}/api/v1/webrtc/voice/offer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('beautyai_access_token')}`
        },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          language: language
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('📥 Received answer from server');
      
      // Store session ID
      sessionIdRef.current = data.session_id || data.peer_id;
      console.log('🔑 Session ID:', sessionIdRef.current);

      // Set remote description
      await pc.setRemoteDescription(new RTCSessionDescription({
        type: 'answer',
        sdp: data.sdp
      }));
      remoteDescriptionSetRef.current = true;
      console.log('✅ Remote description set');
      
      // Send queued local ICE candidates
      if (pendingLocalIceCandidatesRef.current.length > 0 && sessionIdRef.current) {
        console.log(`🧊 Sending ${pendingLocalIceCandidatesRef.current.length} queued local ICE candidates`);
        for (const candidate of pendingLocalIceCandidatesRef.current) {
          try {
            await fetch(`${API_BASE}/api/v1/webrtc/voice/ice`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('beautyai_access_token')}`
              },
              body: JSON.stringify({
                session_id: sessionIdRef.current,
                candidate: candidate.candidate,
                sdp_mid: candidate.sdpMid,
                sdp_m_line_index: candidate.sdpMLineIndex
              })
            });
          } catch (e) {
            console.error('❌ Error sending queued ICE candidate:', e);
          }
        }
        pendingLocalIceCandidatesRef.current = [];
      }

      // Process queued ICE candidates
      await processIceCandidateQueue();

      // Add ICE candidates from server
      if (data.ice_candidates) {
        for (const candidate of data.ice_candidates) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
            console.log('✅ Added server ICE candidate');
          } catch (e) {
            console.error('❌ Error adding server ICE candidate:', e);
          }
        }
      }

      setConnectionState('connecting');
      setIsLoading(false);

      // Track usage after successful connection
      setTimeout(async () => {
        try {
          await guestApi.incrementUsage();
          console.log('✅ Usage tracked');
        } catch (err) {
          console.error('Failed to track usage:', err);
        }
      }, 5000); // Track after 5 seconds of active session

    } catch (error: any) {
      console.error('❌ Error starting session:', error);
      setError(error.message || 'Failed to start session');
      setIsLoading(false);
      stopSession();
    }
  };

  // Stop session
  const stopSession = () => {
    // Close peer connection
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    // Stop local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
      localStreamRef.current = null;
    }

    // Stop audio player
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current = null;
    }

    // Reset state
    dcRef.current = null;
    remoteDescriptionSetRef.current = false;
    iceCandidateQueueRef.current = [];
    sessionIdRef.current = null;
    pendingLocalIceCandidatesRef.current = [];
    setConnectionState('disconnected');
    setVadStatus('🔇 Mic Muted');
    setIsMicMuted(false);
    setIsLoading(false);

    addMessage('system', 'Session ended.');
  };

  // Get status display
  const getStatusDisplay = () => {
    switch (connectionState) {
      case 'connecting':
        return { text: 'Connecting...', color: 'text-yellow-500' };
      case 'connected':
        return { text: 'Connected', color: 'text-green-500' };
      case 'listening':
        return { text: '🎤 Listening...', color: 'text-blue-500' };
      case 'processing':
        return { text: 'Thinking...', color: 'text-purple-500' };
      case 'speaking':
        return { text: '🔊 Speaking...', color: 'text-pink-500' };
      default:
        return { text: 'Disconnected', color: 'text-gray-500' };
    }
  };

  const status = getStatusDisplay();

  if (error && !connectionState) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle className="text-red-600">{t.error}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700 mb-4">{error}</p>
            <Button onClick={() => navigate('/app')} variant="outline">
              {t.backToDashboard}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {language === 'ar' ? 'تجربة المحادثة الصوتية' : 'Voice Conversation Demo'}
          </h1>
          <p className="text-gray-600">
            {language === 'ar' 
              ? 'تحدث بشكل طبيعي وسيتم الرد عليك في الوقت الفعلي'
              : 'Speak naturally and get real-time AI responses'}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main chat area */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${
                    connectionState === 'connected' || connectionState === 'listening' ? 'bg-green-500 animate-pulse' :
                    connectionState === 'processing' ? 'bg-purple-500 animate-pulse' :
                    connectionState === 'speaking' ? 'bg-pink-500 animate-pulse' :
                    connectionState === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                    'bg-gray-400'
                  }`} />
                  <span className={`font-semibold ${status.color}`}>
                    {status.text}
                  </span>
                </div>
                <span className="text-sm text-gray-600">{vadStatus}</span>
              </CardHeader>
              
              <CardContent>
                {/* Chat messages */}
                <div 
                  ref={chatBoxRef}
                  className="h-[400px] overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg space-y-4"
                >
                  {messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`${
                        msg.role === 'system' 
                          ? 'text-center text-sm text-gray-500 py-2'
                          : 'flex flex-col'
                      }`}
                    >
                      {msg.role === 'system' ? (
                        <span className="bg-gray-200 px-3 py-1 rounded-full inline-block">
                          {msg.text}
                        </span>
                      ) : (
                        <>
                          {/* Role label */}
                          <span className={`text-xs font-semibold mb-1 ${
                            msg.role === 'user' 
                              ? 'text-blue-600' 
                              : 'text-emerald-600'
                          }`}>
                            {msg.role === 'user' 
                              ? (language === 'ar' ? '👤 أنت' : '👤 You') 
                              : (language === 'ar' ? '🤖 المساعد' : '🤖 AI Assistant')
                            }
                          </span>
                          {/* Message bubble */}
                          <div
                            className={`p-3 rounded-lg ${
                              msg.role === 'user' 
                                ? 'bg-blue-100 border-l-4 border-blue-500' 
                                : 'bg-emerald-50 border-l-4 border-emerald-500'
                            } ${msg.isRTL ? 'text-right' : 'text-left'}`}
                            dir={msg.isRTL ? 'rtl' : 'ltr'}
                          >
                            {msg.text}
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                  
                  {/* Current assistant message (streaming) */}
                  {currentAssistantMessage && (
                    <div className="flex flex-col">
                      <span className="text-xs font-semibold mb-1 text-emerald-600">
                        {language === 'ar' ? '🤖 المساعد' : '🤖 AI Assistant'}
                      </span>
                      <div
                        className={`p-3 rounded-lg bg-emerald-50 border-l-4 border-emerald-500 ${
                          isArabicText(currentAssistantMessage) ? 'text-right' : 'text-left'
                        }`}
                        dir={isArabicText(currentAssistantMessage) ? 'rtl' : 'ltr'}
                      >
                        {currentAssistantMessage}
                        <span className="inline-block w-2 h-4 ml-1 bg-emerald-400 animate-pulse" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="flex items-center gap-3">
                  <Select
                    value={language}
                    onValueChange={(value: 'ar' | 'en') => setLanguage(value)}
                    disabled={connectionState !== 'disconnected'}
                  >
                    <SelectTrigger className="w-[140px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ar">العربية</SelectItem>
                      <SelectItem value="en">English</SelectItem>
                    </SelectContent>
                  </Select>

                  {connectionState === 'disconnected' ? (
                    <Button
                      onClick={startSession}
                      disabled={isLoading || !!error}
                      className="flex-1"
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          {t.connecting}
                        </>
                      ) : (
                        <>
                          <Phone className="mr-2 h-4 w-4" />
                          {language === 'ar' ? 'بدء المحادثة' : 'Start Conversation'}
                        </>
                      )}
                    </Button>
                  ) : (
                    <Button
                      onClick={stopSession}
                      variant="destructive"
                      className="flex-1"
                    >
                      <PhoneOff className="mr-2 h-4 w-4" />
                      {language === 'ar' ? 'إنهاء المحادثة' : 'End Conversation'}
                    </Button>
                  )}

                  {connectionState !== 'disconnected' && (
                    <Button
                      onClick={() => handleMicControl(!isMicMuted)}
                      variant="outline"
                      size="icon"
                    >
                      {isMicMuted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Metrics panel */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {language === 'ar' ? 'مقاييس الأداء' : 'Performance Metrics'}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-gray-600">
                    {language === 'ar' ? 'الكلمات في الثانية' : 'Tokens/Second'}
                  </div>
                  <div className="text-2xl font-bold text-blue-600">
                    {metrics.tps?.toFixed(2) || '--'}
                  </div>
                </div>

                <div>
                  <div className="text-sm text-gray-600">
                    {language === 'ar' ? 'زمن الاستجابة' : 'LLM Latency'}
                  </div>
                  <div className="text-2xl font-bold text-purple-600">
                    {metrics.llm_latency?.toFixed(2) || '--'}s
                  </div>
                </div>

                <div>
                  <div className="text-sm text-gray-600">
                    {language === 'ar' ? 'وقت التعرف على الصوت' : 'STT Time'}
                  </div>
                  <div className="text-2xl font-bold text-green-600">
                    {metrics.stt_time?.toFixed(2) || '--'}s
                  </div>
                </div>

                <div>
                  <div className="text-sm text-gray-600">
                    {language === 'ar' ? 'وقت توليد الصوت' : 'TTS Time'}
                  </div>
                  <div className="text-2xl font-bold text-pink-600">
                    {metrics.tts_time?.toFixed(2) || '--'}s
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <div className="text-xs text-gray-500">
                    {language === 'ar' ? 'حالة الاتصال' : 'Connection State'}
                  </div>
                  <div className="text-sm font-medium mt-1">
                    {connectionState}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
