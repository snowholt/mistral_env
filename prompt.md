
# Planning Prompt: Integrate BeautyAI into gmai.sa Website (Do Not Break Landing Page)

You are a senior full‑stack architect and delivery lead. Your job is to produce a clear, manager-friendly but technically correct implementation plan to integrate **BeautyAI** (WhatsApp automation + web chatbot + client/admin dashboards) into our existing production website.

## 1) Hard Context You Must Use (from this repo)

### Website (the product site we will integrate into)
- Location (extracted snapshot): `_website_snapshot/gmai.sa/gmai.sa/`
- Tech: **Vite + React + TypeScript + shadcn-ui + Tailwind**
- Routing:
	- `src/App.tsx` currently defines only:
		- `/` → landing page (`src/pages/Index.tsx`)
		- `/privacy-policy`, `/terms`, and `*` NotFound
- Landing page composition (must be preserved): `src/pages/Index.tsx` uses `Header`, `Hero`, `CaseStudies`, `Services`, `About`, `Contact`, `Footer`.
- Important constraint: the website is deployed as a static SPA (dist upload); routing requires server fallback to `index.html`.

### Backend (BeautyAI Platform)
- FastAPI app: `backend/src/beautyai_inference/api/app.py`
- Existing WhatsApp SaaS endpoints (already implemented):
	- Auth (JWT): `/api/v1/whatsapp/auth/*` (register/login/refresh/me/etc.)
	- WhatsApp manager: `/api/v1/whatsapp/*` (customers, Meta embedded signup, agent config, inbox APIs)
	- Webhook: `/api/v1/whatsapp/webhook`
	- Inbox realtime: WebSocket `/api/v1/whatsapp/inbox/ws?token=...`
- Existing LLM chat endpoint (for web widget / internal usage): `POST /inference/chat`
- Existing DB models for multi-tenant WhatsApp manager: `backend/src/beautyai_inference/database/models.py` (User, Customer, WhatsAppAccount, AgentConfig, Conversation, Message).

### Legacy/Reference UI (not the target UI)
- There are existing Flask templates for WhatsApp management (reference only): `frontend/src/templates/whatsapp/`

## 2) Business Goal (say it simply)
We will keep the current marketing/landing website as-is, and **add**:
- Login/Signup
- **Customer dashboard** for configuring WhatsApp + AI instructions (RAG/FAQ/company info), testing, and going live
- **Admin dashboard** for managing customers, usage, billing/income, and operational controls
- A **website chatbot widget** (floating icon → popup chat) powered by our **Genius AI LLM**, so our customers can embed/enable it on their websites, and end-users receive customer service responses.

## 3) Non‑Negotiable Constraints
1. Do **not** redesign or break the existing landing experience.
	 - Treat landing page UI/sections (Hero, etc.) as “read-only”.
	 - Only allow minimal, non-breaking additions (example: adding a Login link/button to the existing header).
2. Integration happens inside the existing website codebase (React SPA) — not the Flask `frontend/` templates.
3. Must support production hosting on Alibaba Cloud (static site + backend APIs behind domain/Nginx).
4. Must be multi-tenant: each customer/business has isolated configuration, WhatsApp account(s), and knowledge.
5. Security first: JWT auth, RBAC (client vs admin), CORS, secret management.

## 4) What You Must Produce (the planning deliverables)
Deliver a planning document in Markdown with these outputs:

### A) Executive Summary (for managers)
- 8–12 bullets, no jargon.
- What users can do, what admins can do, what we will ship first.

### B) User Journeys (simple)
- Client onboarding journey (signup → create business → connect WhatsApp → configure AI → test → go live)
- End user journey (customer’s customer sends WhatsApp message → AI responds)
- Admin journey (view customers/usage/revenue → charge accounts → manage plans)

### C) System Architecture (Mermaid diagrams)
Include Mermaid blocks for:
1. **High-level system flow** (Client Dashboard → Backend → WhatsApp/Meta → Genius AI LLM → End User)
2. **Sequence diagram** for a WhatsApp inbound message handled by webhook → stored → LLM response → outgoing message
3. **Route map** (public site vs authenticated customer area vs admin area)
4. Optional: simple ER diagram for SaaS entities (User, Customer, WhatsAppAccount, AgentConfig, KnowledgeBase, UsageEvent, Subscription)

### D) Website Integration Plan (React)
Provide:
- Proposed new routes (examples):
	- `/auth/login`, `/auth/register`
	- `/app/*` (customer dashboard)
	- `/admin/*` (admin dashboard)
- Component/layout approach that preserves landing pages.
- How to add the floating chatbot widget globally without breaking existing pages.
- State management approach (React Query + token storage + role routing guards).
- UI reuse guidance (shadcn components, existing header button commented out).

### E) Backend/API Plan
Provide:
- Which existing endpoints will be used as-is (WhatsApp auth/manager/inbox ws).
- Which new endpoints are needed (if any), for:
	- Knowledge base (RAG/FAQ uploads, CRUD, indexing)
	- Web chat widget session + messaging (multi-tenant)
	- Usage metering, plans/subscriptions, billing events
	- Admin reporting endpoints
- Authentication alignment:
	- Use WhatsApp JWT for dashboards and (optionally) for protected webchat admin APIs
	- Define a safe approach for the public webchat widget (e.g., widget token per customer, rate limits)

### F) Data Model Extensions
Starting from existing WhatsApp models, propose additions for:
- Subscription/Plan
- UsageEvent (tokens/messages/minutes)
- KnowledgeBase + Document + Chunk/Embedding index metadata
- Customer API keys / Widget tokens

### G) Deployment Plan (Alibaba Cloud)
Must include:
- Static hosting route fallback for SPA
- API base URL strategy (env var like `VITE_API_BASE_URL`)
- CORS and HTTPS
- Nginx reverse proxy concept and WebSocket support
- Secrets handling (Meta credentials, JWT secret, DB URL)

### H) Phasing + Timeline
Propose a realistic phased rollout:
1. Phase 1: Auth + Customer dashboard skeleton + WhatsApp onboarding + agent config + inbox (MVP)
2. Phase 2: Web chat widget MVP + demo chat inside dashboard
3. Phase 3: Knowledge base (RAG) + usage metering
4. Phase 4: Billing + admin controls + automation

For each phase include: scope, acceptance criteria, and main risks.

### I) Testing & Acceptance
Provide:
- Key E2E tests (login, connect WhatsApp, receive message, AI response)
- Security checks (tenant isolation, token expiry, webhook verification)
- Performance checks (LLM latency, inbox WS stability)

## 5) Rules for Your Output
- Do **not** write code. This is a plan only.
- Keep language clear; define any necessary terms once.
- When you’re unsure, write assumptions explicitly and list questions.
- Ask up to 10 clarifying questions at the end, ordered by impact.

## 6) Extra Inputs You Should Consider (will be shared)
- A “new feature spec” will be shared after your first plan draft. Leave placeholders for it, and show where it plugs into the phases.

---

Now produce the full planning document.

