# 🤖 AI Chatbot Web Widget - Status Report

**Date**: December 22, 2025  
**Status**: ⚠️ **DEMO MODE** - Not Connected to Real AI Model

---

## 📊 Executive Summary

The AI chatbot widget currently visible on `https://portal.gmai.sa` is running in **DEMO MODE** with hardcoded responses. It is **NOT** connected to the Qwen3-14B model or any real AI backend.

### Current State
- ✅ Widget UI fully implemented and visible on website
- ✅ Backend API endpoints ready and functional
- ✅ Qwen3-14B model loaded and ready in backend
- ❌ Widget using `widgetToken="demo"` - bypassing real AI
- ❌ No widget tokens created in database
- ❌ Demo responses hardcoded in frontend component

---

## 🔌 Connection Architecture

### How It SHOULD Work (Production Flow)

```
┌─────────────────┐
│  Website User   │
└────────┬────────┘
         │ 1. Opens chatbot
         ▼
┌─────────────────────────────────────┐
│  ChatWidget.tsx (Frontend)          │
│  - widgetToken: "real_token_here"   │
└─────────────────┬───────────────────┘
                  │ 2. POST /api/v1/webchat/session
                  ▼
┌─────────────────────────────────────┐
│  webchat.py (Backend API)           │
│  - Validates widget token           │
│  - Creates chat session             │
│  - Returns session_token            │
└─────────────────┬───────────────────┘
                  │ 3. Returns session
                  ▼
┌─────────────────────────────────────┐
│  User sends message                 │
└─────────────────┬───────────────────┘
                  │ 4. POST /api/v1/webchat/message
                  ▼
┌─────────────────────────────────────┐
│  webchat.py - send_message()        │
│  - Validates session                │
│  - Gets customer's agent config     │
│  - Builds conversation history      │
└─────────────────┬───────────────────┘
                  │ 5. Calls generate_ai_response()
                  ▼
┌─────────────────────────────────────┐
│  generate_ai_response()             │
│  - Gets system prompt from agent    │
│  - Calls InferenceService.chat()    │
└─────────────────┬───────────────────┘
                  │ 6. Uses inference endpoint
                  ▼
┌─────────────────────────────────────┐
│  inference.py - chat_completion()   │
│  - Uses PersistentModelManager      │
│  - Gets persistent Qwen3 model      │
│  - Generates response               │
└─────────────────┬───────────────────┘
                  │ 7. Returns AI response
                  ▼
┌─────────────────────────────────────┐
│  qwen3-unsloth-q4ks Model           │
│  - Llama.cpp engine                 │
│  - 14B parameters (Q4_K_S)          │
│  - Loaded at: /home/lumi/.cache/... │
└─────────────────────────────────────┘
```

### How It CURRENTLY Works (Demo Mode)

```
┌─────────────────┐
│  Website User   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  ChatWidget.tsx (Frontend)          │
│  - widgetToken: "demo" ❌           │
│  - Demo mode detected               │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Hardcoded Demo Responses:          │
│  - "شكراً لتواصلك معنا! 🌟"         │
│  - "نعم، يمكننا مساعدتك..."        │
│  - "خدماتنا تشمل أتمتة..."         │
│  - Random selection on each msg     │
└─────────────────────────────────────┘

❌ NO CONNECTION TO QWEN3 MODEL
❌ NO BACKEND API CALLS
```

---

## 🔍 Technical Details

### Frontend Configuration
**File**: `_website_snapshot/gmai.sa/gmai.sa/src/App.tsx`

```tsx
<ChatWidget
  widgetToken="demo"  // ❌ PROBLEM: Using demo token
  apiUrl={import.meta.env.VITE_API_URL || 'https://api.gmai.sa'}
  primaryColor="#0ea5e9"
  headerText="مساعد الذكاء الاصطناعي"
  placeholderText="اكتب رسالتك..."
  welcomeMessage="مرحباً! 👋 كيف يمكنني مساعدتك اليوم؟"
  position="bottom-right"
/>
```

### Demo Mode Logic
**File**: `_website_snapshot/gmai.sa/gmai.sa/src/components/ChatWidget.tsx`

```tsx
const isDemoMode = config.widgetToken === 'demo';

// When sending message:
if (isDemoMode) {
  // Simulates 800-1500ms delay
  await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));
  
  // Returns random hardcoded response
  const demoResponses = [
    "شكراً لتواصلك معنا! 🌟 يسعدنا مساعدتك. كيف يمكنني خدمتك؟",
    "نعم، يمكننا مساعدتك في ذلك! هل تود معرفة المزيد عن خدماتنا؟",
    // ... 3 more responses
  ];
}
```

---

## 🔐 Authentication & Token System

### Widget Token Purpose
Widget tokens are **API keys** for embedding the chatbot on customer websites. They:

1. **Authenticate** customer's website to use the chatbot
2. **Link** conversations to the correct customer account
3. **Track usage** for billing and analytics
4. **Apply** customer-specific AI agent configuration
5. **Enforce** rate limits per customer

### Token Validation Flow

```python
# backend/src/beautyai_inference/api/endpoints/webchat.py

async def validate_widget_token(token, request, db, redis):
    """
    1. Hashes the token for secure lookup
    2. Checks Redis cache first (fast)
    3. Falls back to database query
    4. Validates:
       - Token exists and is active
       - Token not expired
       - Domain is whitelisted (if set)
       - Rate limits not exceeded
    5. Returns: (WidgetToken, Customer)
    """
```

### Current Database Status
```bash
$ Check widget tokens in database
Result: No widget tokens found in database
```

**Implication**: Even if you change from `"demo"` to a real token, there are no tokens in the database yet!

---

## 🎯 What Needs to Happen for Real AI Connection

### Step 1: Create Widget Token in Database

You need to create a widget token for your customer account. This can be done through:

**Option A**: Admin dashboard (once implemented)  
**Option B**: Direct database insert  
**Option C**: Python script

Example script to create a widget token:

```python
# Create token for customer (e.g., customer_id=1)
from beautyai_inference.database.models import WidgetToken, Customer

async def create_widget_token(customer_id: int):
    widget = WidgetToken(
        customer_id=customer_id,
        token=WidgetToken.generate_token(),  # Generates secure token
        name="Portal Website Widget",
        allowed_domains=["portal.gmai.sa", "gmai.sa"],  # Whitelist
        is_active=True
    )
    db.add(widget)
    await db.commit()
    print(f"Created widget token: {widget.token}")
```

### Step 2: Update Frontend Configuration

Replace the demo token in `App.tsx`:

```tsx
// OLD (Demo Mode):
widgetToken="demo"

// NEW (Real AI):
widgetToken="wgt_abc123xyz456..."  // Real token from database
```

### Step 3: Configure Agent Settings

Ensure the customer account has an `AgentConfig` with:
- **system_prompt**: Instructions for AI behavior
- **model_name**: "qwen3-unsloth-q4ks" (to use Qwen3-14B)
- **temperature**: 0.7 (recommended)
- **max_tokens**: 500 (for chat responses)

### Step 4: Test the Integration

```bash
# 1. Create session
curl -X POST https://api.gmai.sa/api/v1/webchat/session \
  -H "Content-Type: application/json" \
  -d '{
    "widget_token": "wgt_abc123xyz456...",
    "page_url": "https://portal.gmai.sa"
  }'

# 2. Send message
curl -X POST https://api.gmai.sa/api/v1/webchat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_token": "session_token_from_step_1",
    "message": "مرحبا، كيف حالك؟"
  }'
```

---

## 🧠 Model Integration Details

### Does Webchat Use Qwen3-14B?
**YES** - When properly configured with real widget token, it will use:

```
Model: qwen3-unsloth-q4ks
Engine: llama.cpp
Size: 14B parameters (Q4_K_S quantization)
Location: /home/lumi/.cache/huggingface/hub/models--unsloth--Qwen3-14B-GGUF/
Status: ✅ Loaded and ready
```

### Inference Flow with Real Token

```python
# webchat.py: send_message() calls:
await generate_ai_response(customer, conversation_history)

# generate_ai_response() calls:
from ...services.inference import InferenceService
inference_service = InferenceService.get_instance()
response = await inference_service.chat(messages, max_tokens=500, temperature=0.7)

# InferenceService uses:
# 1. PersistentModelManager.get_llm_model() 
#    → Returns preloaded Qwen3-14B instance
# 2. Generates response using llama.cpp
# 3. Returns: (response_text, input_tokens, output_tokens)
```

### Does It Need Authentication?
**Widget Token = Authentication**

- ❌ **NO JWT/User Login**: Chatbot works without user accounts
- ✅ **Widget Token**: Acts as API key authentication
- ✅ **Session Token**: Created after widget token validation
- ✅ **Rate Limiting**: 60 requests/minute per IP address

---

## 📈 Benefits of Connecting to Real AI

### Current Demo Mode Limitations
- ❌ Only 5 hardcoded responses (repetitive)
- ❌ No context awareness (can't remember conversation)
- ❌ No personalization per customer
- ❌ No Arabic language model optimization
- ❌ Can't answer specific questions
- ❌ No usage tracking or analytics

### With Real Qwen3-14B Connection
- ✅ **Context-aware**: Remembers last 20 messages
- ✅ **Arabic-optimized**: Fine-tuned for Arabic conversations
- ✅ **Personalized**: Uses customer's agent configuration
- ✅ **Intelligent**: Can answer complex questions
- ✅ **Tracked**: Input/output tokens for billing
- ✅ **Customizable**: Temperature, max_tokens, system prompt
- ✅ **Scalable**: Handles concurrent sessions efficiently

---

## 🚀 Quick Fix Action Plan

### Immediate Steps (15 minutes)

1. **Create Widget Token**
   ```bash
   cd /home/lumi/beautyai/backend
   source venv/bin/activate
   python scripts/create_widget_token.py --customer-id 1
   ```

2. **Update Frontend**
   ```bash
   cd /home/lumi/beautyai/_website_snapshot/gmai.sa/gmai.sa
   # Edit src/App.tsx - replace "demo" with real token
   npm run build
   ```

3. **Deploy Frontend**
   ```bash
   sudo cp -r dist/* /var/www/portal.gmai.sa/html/
   ```

4. **Test Integration**
   - Open https://portal.gmai.sa
   - Click chatbot icon
   - Send message: "مرحبا"
   - Expect: Real AI response (not demo response)

### Verification Checklist
- [ ] Widget token exists in database
- [ ] `widgetToken` in App.tsx updated to real token
- [ ] Frontend rebuilt and deployed
- [ ] Chatbot sends API requests to backend
- [ ] Backend generates responses using Qwen3-14B
- [ ] Conversation history maintained in database
- [ ] Usage tokens tracked in `usage_events` table

---

## 🔧 Troubleshooting

### If Chatbot Still Shows Demo Responses
1. Check browser console for errors
2. Verify `widgetToken` value in App.tsx
3. Clear browser cache and reload
4. Check backend logs: `sudo journalctl -u beautyai-api.service -f`

### If API Returns 401/403 Errors
- Widget token not found or inactive in database
- Token doesn't match customer account
- Domain not whitelisted in `allowed_domains`

### If AI Responses Are Slow
- Check GPU utilization: `nvidia-smi`
- Verify model is loaded: `curl http://localhost:8000/models/loaded`
- Increase `max_tokens` limit in agent config

---

## 📝 Related Files

### Frontend
- `_website_snapshot/gmai.sa/gmai.sa/src/App.tsx` - Widget initialization
- `_website_snapshot/gmai.sa/gmai.sa/src/components/ChatWidget.tsx` - Widget component

### Backend
- `backend/src/beautyai_inference/api/endpoints/webchat.py` - API endpoints
- `backend/src/beautyai_inference/api/endpoints/inference.py` - LLM inference
- `backend/src/beautyai_inference/core/persistent_model_manager.py` - Model loading

### Database
- `beautyai.widget_tokens` - Widget authentication tokens
- `beautyai.webchat_sessions` - Active chat sessions
- `beautyai.webchat_messages` - Conversation history
- `beautyai.agent_configs` - Customer AI configurations

---

## 🎯 Conclusion

**Current Status**: Demo Mode (Fake AI)  
**Target Status**: Real AI with Qwen3-14B  
**Blocker**: No widget token created  
**Effort to Fix**: ~15 minutes  
**Impact**: Transform from demo to production-ready AI chatbot

The infrastructure is **100% ready**:
- ✅ Backend API functional
- ✅ Qwen3-14B model loaded
- ✅ Database schema ready
- ✅ Frontend UI polished

Only missing: **Real widget token** instead of `"demo"`

---

**Report Generated**: December 22, 2025  
**BeautyAI Platform Version**: 2.0.0  
**Model**: Qwen3-14B-Q4_K_S (Loaded & Ready)
