You are acting as a senior backend architect and platform engineer.

Please analyze the entire codebase and project structure, with a focus on how WhatsApp, Meta APIs, and business entities are currently modeled and integrated.

Context:
- This is a multi-tenant B2B SaaS platform.
- We provide WhatsApp AI bots for businesses.
- Each business connects its own WhatsApp Business Account (WABA) via Meta Embedded Signup.
- The platform includes:
  - Admin dashboard (platform owner)
  - Business dashboard (our customers)
  - WhatsApp inbox, AI agents, automation, billing
- The backend is responsible for WhatsApp Cloud API integration, webhooks, AI routing, and account management.

Goals of this review:
1. Identify how “business”, “customer”, “WhatsApp account”, and “user” are currently represented in the codebase.
2. Detect architectural gaps or coupling issues related to:
   - Meta Business Manager
   - WhatsApp Business Accounts (WABA)
   - Roles, ownership, and permissions
3. Evaluate whether the current structure can support a real Meta Business Management integration (not just a test API call).
4. Propose a clean, scalable architecture to:
   - Introduce a first-class Business domain object
   - Map Meta Business Manager → Internal Business → WABA → AI Agents
   - Support future features like role-based access, lifecycle management, billing, and audits
5. Recommend concrete changes:
   - New modules or services (if needed)
   - Data models / schemas
   - API boundaries
   - Sync or verification flows with Meta Graph API
6. Highlight what can remain as-is vs what should be refactored.
7. Suggest a minimal MVP implementation plan followed by an ideal long-term architecture.

Please be specific, reference actual folders/files when relevant, and think in terms of production-grade SaaS systems.




I'll conduct a comprehensive architecture review. Let me start by gathering deep context about the current data models, services, and API structure. 

Architecture codebase review

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/database/models.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/whatsapp_manager.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/whatsapp_webhook.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/whatsapp_auth.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/admin.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/billing.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/endpoints/dashboard.py)

Read [](file:///home/lumi/beautyai/backend/alembic/versions)

Searched for regex `Business|BusinessManager|business_id` (`**/*.py`), 20 results

Read [](file:///home/lumi/beautyai/backend/alembic/versions/20241219_001_initial_saas_platform_schema.py)

Searched for files matching `**/billing*.py`, 1 match

Searched for regex `graph\.facebook|META_API|meta_api|whatsapp.*api`, 20 results

Searched for regex `class StripeService|def get_stripe_service|handle_stripe_webhook`, 8 results

Searched for regex `include_router|APIRouter\(prefix` (`**/app.py`), 20 results

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/services/billing/stripe_service.py)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/api/app.py#200-200), lines 200 to 400

Read [](file:///home/lumi/.vscode-server-insiders/data/User/workspaceStorage/59a2628ff9c71e163531132839d5e844/GitHub.copilot-chat/chat-session-resources/4cc5ad44-8111-48c3-94b2-b99d7ef134d3/toolu_01E1sHmUmoRZdxQb26DNK5Qw__vscode-1770210529646/content.txt)

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/database/models.py#1-1), lines 1 to 201

Read [](file:///home/lumi/beautyai/backend/src/beautyai_inference/database/models.py#300-300), lines 300 to 600

Now I have a complete picture. Let me compile the comprehensive architecture review.

---

# Architecture Review: BeautyAI WhatsApp SaaS Platform

## Executive Summary

The current architecture is a **functional MVP** designed for single-user-per-business WhatsApp automation. However, it has significant gaps that prevent it from scaling to a production-grade multi-tenant B2B SaaS with proper Meta Business Manager integration.

**Key findings:**
- ✅ Solid foundation: FastAPI + SQLAlchemy + PostgreSQL + pgvector
- ✅ Good separation: API endpoints, services, database models
- ⚠️ Naming confusion: "Customer" means "Business/Tenant", not CRM contact
- 🔴 Security risk: Meta access tokens stored in plaintext
- 🔴 Missing: Team collaboration, RBAC, audit logging
- 🔴 Missing: First-class Business domain entity aligned with Meta Business Manager

---

## 1. Current Entity Model

### Text-Based ERD

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CURRENT STATE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

    User (Platform Account)
    ├── role: user | admin | guest
    ├── stripe_customer_id
    │
    └──1:N──► Customer (ACTUALLY = Business/Tenant)
              ├── name, email, timezone
              ├── widget_* settings
              │
              ├──1:1──► AgentConfig
              │         ├── business_name, tone, system_prompt
              │         ├── services[], products[], locations[], promotions[]
              │         └── wizard_completed, ai_enabled
              │
              ├──1:1──► Subscription
              │         ├── plan_id, stripe_subscription_id
              │         └── messages_used, tokens_used
              │
              ├──1:N──► WhatsAppAccount
              │         ├── phone_number_id, waba_id
              │         ├── access_token (PLAINTEXT!)
              │         │
              │         └──1:N──► Conversation
              │                   ├── contact_phone, contact_name
              │                   └──1:N──► Message
              │
              ├──1:N──► KnowledgeBase ──► Document ──► Chunk (pgvector)
              │
              └──1:N──► WidgetToken ──► WebChatSession ──► WebChatMessage
```

### Current Schema Locations

| Entity | File | Line |
|--------|------|------|
| User | models.py | 163 |
| Customer | models.py | 300 |
| WhatsAppAccount | models.py | 342 |
| AgentConfig | models.py | 390 |
| Conversation | models.py | ~500 |
| Message | models.py | ~550 |
| Subscription | models.py | ~650 |
| Plan | models.py | ~620 |

---

## 2. Identified Architectural Gaps

### 🔴 Critical Issues

| # | Issue | Current State | Risk |
|---|-------|---------------|------|
| 1 | **Plaintext access_token** | `WhatsAppAccount.access_token` is `Text` | Token theft = full WABA compromise |
| 2 | **No token refresh** | No mechanism to refresh expired Meta tokens | Silent failures after token expiry |
| 3 | **No audit logging** | Only OTP verification logs | Compliance failure, no forensics |
| 4 | **Single-user per business** | `Customer.user_id` is N:1 | No team collaboration |

### 🟠 Multi-Tenancy Design Issues

| # | Gap | Current State | Impact |
|---|-----|---------------|--------|
| 5 | **No Team/Org model** | User→Customer is 1:N, but Customer→User is N:1 | Can't have multiple team members |
| 6 | **No granular RBAC** | Only `user/admin/guest` roles | No business-level roles (owner/manager/agent) |
| 7 | **"Customer" naming** | "Customer" = Business/Tenant | Confusing; conflicts with CRM "customer" concept |
| 8 | **Billing architecture** | Subscription 1:1 Customer | What if user has multiple businesses? |

### 🟡 Meta Integration Gaps

| # | Gap | Current State | Needed |
|---|-----|---------------|--------|
| 9 | **No Business entity** | `waba_id` is a field on WhatsAppAccount | First-class entity for Meta Business Manager |
| 10 | **No template management** | Not tracked | Message templates for campaigns |
| 11 | **No quality tracking** | Not stored | Meta quality ratings for account health |
| 12 | **No `business_management` scope** | Only `whatsapp_business_management` + `whatsapp_business_messaging` | Can't read Business Manager assets |

---

## 3. Proposed Architecture

### 3.1 New Domain Model

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PROPOSED ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

    User (Platform Account)
    ├── email, password_hash, is_platform_admin
    │
    └──M:N──► OrganizationMember (pivot table)
              ├── role: owner | admin | member | viewer
              │
              └──► Organization (replaces Customer)
                   ├── name, slug (unique), billing_email
                   ├── meta_business_id (from Business Manager)
                   │
                   ├──1:1──► Subscription
                   │
                   ├──1:1──► AgentConfig
                   │
                   ├──1:N──► MetaCredential (encrypted vault)
                   │         ├── credential_type: user_token | system_user_token
                   │         ├── encrypted_value, key_version
                   │         └── expires_at, scopes[]
                   │
                   ├──1:N──► WhatsAppBusinessAccount (WABA)
                   │         ├── waba_id, name, quality_rating
                   │         ├── credential_id (FK to MetaCredential)
                   │         │
                   │         └──1:N──► WhatsAppPhoneNumber
                   │                   ├── phone_number_id, display_phone_number
                   │                   ├── verified_name, quality_rating
                   │                   │
                   │                   └──1:N──► Conversation
                   │                             └──1:N──► Message
                   │
                   ├──1:N──► Contact (CRM)
                   │         ├── phone, name, email
                   │         ├── tags[], custom_fields{}
                   │         └──1:N──► Conversation
                   │
                   ├──1:N──► MessageTemplate
                   │         ├── meta_template_id, name, status
                   │         └── components[]
                   │
                   └──1:N──► AuditLog
                             ├── user_id, action, resource_type
                             └── resource_id, metadata, ip_address
```

### 3.2 New Entities

#### Organization (replaces Customer)

```python
class Organization(Base):
    """
    Top-level business entity (tenant).
    Maps to Meta Business Manager when connected.
    """
    __tablename__ = "organizations"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    billing_email: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Meta Business Manager integration
    meta_business_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    meta_business_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Relationships
    members: Mapped[List["OrganizationMember"]] = relationship(...)
    wabas: Mapped[List["WhatsAppBusinessAccount"]] = relationship(...)
    credentials: Mapped[List["MetaCredential"]] = relationship(...)
```

#### OrganizationMember (team management)

```python
class OrgRole(enum.Enum):
    OWNER = "owner"      # Full control, billing
    ADMIN = "admin"      # Manage team, settings
    MEMBER = "member"    # Use inbox, respond
    VIEWER = "viewer"    # Read-only

class OrganizationMember(Base):
    """Team membership with role-based access."""
    __tablename__ = "organization_members"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    role: Mapped[OrgRole] = mapped_column(StringEnumType(OrgRole))
    invited_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    joined_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

#### MetaCredential (encrypted vault)

```python
class MetaCredential(Base):
    """
    Encrypted storage for Meta API tokens.
    Supports token rotation and multiple credential types.
    """
    __tablename__ = "meta_credentials"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    
    credential_type: Mapped[str] = mapped_column(String(50))  # user_token, system_user_token
    encrypted_value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_key_version: Mapped[int] = mapped_column(Integer, default=1)
    
    scopes: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

#### WhatsAppBusinessAccount (WABA)

```python
class WhatsAppBusinessAccount(Base):
    """
    WhatsApp Business Account from Meta.
    One WABA can have multiple phone numbers.
    """
    __tablename__ = "whatsapp_business_accounts"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    credential_id: Mapped[int] = mapped_column(ForeignKey("meta_credentials.id"))
    
    waba_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Meta quality/health metrics
    quality_rating: Mapped[Optional[str]] = mapped_column(String(20))  # GREEN, YELLOW, RED
    messaging_limit_tier: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Webhook configuration
    webhook_verify_token: Mapped[str] = mapped_column(String(255))
    
    # Relationships
    phone_numbers: Mapped[List["WhatsAppPhoneNumber"]] = relationship(...)
    templates: Mapped[List["MessageTemplate"]] = relationship(...)
```

#### WhatsAppPhoneNumber (replaces WhatsAppAccount)

```python
class WhatsAppPhoneNumber(Base):
    """Individual phone number within a WABA."""
    __tablename__ = "whatsapp_phone_numbers"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    waba_id: Mapped[int] = mapped_column(ForeignKey("whatsapp_business_accounts.id"))
    
    phone_number_id: Mapped[str] = mapped_column(String(100), unique=True)
    display_phone_number: Mapped[str] = mapped_column(String(50))
    verified_name: Mapped[Optional[str]] = mapped_column(String(255))
    quality_rating: Mapped[Optional[str]] = mapped_column(String(20))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

#### Contact (CRM foundation)

```python
class Contact(Base):
    """
    WhatsApp contact / end-user.
    Separates contact identity from conversation threads.
    """
    __tablename__ = "contacts"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    email: Mapped[Optional[str]] = mapped_column(String(255))
    
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    custom_fields: Mapped[dict] = mapped_column(JSON, default={})
    
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Unique per org
    __table_args__ = (UniqueConstraint('organization_id', 'phone'),)
```

#### AuditLog (compliance)

```python
class AuditLog(Base):
    """Security and compliance audit trail."""
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[Optional[int]] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    
    action: Mapped[str] = mapped_column(String(100))  # e.g., "whatsapp.account.connected"
    resource_type: Mapped[str] = mapped_column(String(100))
    resource_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    metadata: Mapped[dict] = mapped_column(JSON, default={})
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

---

## 4. What Can Remain As-Is vs. Refactor

### ✅ Keep As-Is

| Component | Reason |
|-----------|--------|
| User model | Good foundation, just needs `is_platform_admin` flag |
| AgentConfig + wizard entities | Well-designed, fully functional |
| Conversation + Message | Solid structure |
| Knowledge Base (RAG) | pgvector integration is good |
| Web Chat widget system | Clean separation |
| Billing models (Plan, Subscription, UsageEvent) | Adequate for MVP |
| API endpoint structure | FastAPI patterns are clean |

### 🔄 Refactor

| Component | Change | Priority |
|-----------|--------|----------|
| `Customer` → `Organization` | Rename + add team relationships | High |
| `WhatsAppAccount` → `WhatsAppPhoneNumber` | Add WABA parent entity | High |
| `access_token` storage | Move to encrypted `MetaCredential` vault | Critical |
| Add `OrganizationMember` | Team collaboration | High |
| Add `Contact` entity | CRM foundation | Medium |
| Add `AuditLog` | Compliance | High |
| Add `MessageTemplate` | WhatsApp campaigns | Medium |

---

## 5. API Boundary Changes

### Current Endpoints (keep)

| Endpoint Group | Status |
|----------------|--------|
| `/api/v1/auth/*` | Keep, extend for org context |
| `/api/v1/whatsapp/webhook` | Keep (public webhook) |
| `/api/v1/billing/*` | Keep, add org context |
| `/api/v1/kb/*` | Keep, add org context |
| `/api/v1/webchat/*` | Keep |

### New Endpoints (add)

```
/api/v1/organizations/
├── GET    /                      # List user's orgs
├── POST   /                      # Create org
├── GET    /{org_id}              # Org details
├── PATCH  /{org_id}              # Update org
├── DELETE /{org_id}              # Delete org (owner only)
│
├── GET    /{org_id}/members      # List team members
├── POST   /{org_id}/members      # Invite member
├── DELETE /{org_id}/members/{id} # Remove member
├── PATCH  /{org_id}/members/{id} # Change role
│
├── GET    /{org_id}/wabas        # List WABAs
├── POST   /{org_id}/wabas        # Connect WABA (Embedded Signup)
├── GET    /{org_id}/wabas/{id}   # WABA details
├── GET    /{org_id}/wabas/{id}/phones  # Phone numbers
│
├── GET    /{org_id}/contacts     # List contacts
├── GET    /{org_id}/contacts/{id}
├── PATCH  /{org_id}/contacts/{id}
│
└── GET    /{org_id}/audit-logs   # Audit trail (admin)
```

### Refactored Endpoints

```
# BEFORE (implicit org context via user)
GET /api/v1/whatsapp/accounts

# AFTER (explicit org context)
GET /api/v1/organizations/{org_id}/wabas/{waba_id}/phones
```

---

## 6. New Services Needed

### MetaCredentialService

```python
# backend/src/beautyai_inference/services/meta_credential.py

class MetaCredentialService:
    """Encrypted credential vault for Meta API tokens."""
    
    async def store_token(self, org_id: int, token: str, scopes: list[str], expires_at: datetime) -> MetaCredential:
        """Encrypt and store a new Meta token."""
        
    async def get_token(self, credential_id: int) -> str:
        """Decrypt and return token (with audit logging)."""
        
    async def refresh_token(self, credential_id: int) -> str:
        """Exchange refresh token for new access token."""
        
    async def revoke_token(self, credential_id: int) -> None:
        """Revoke and delete token."""
```

### MetaGraphService

```python
# backend/src/beautyai_inference/services/meta_graph.py

class MetaGraphService:
    """Meta Graph API client with proper error handling."""
    
    async def get_businesses(self, token: str) -> list[dict]:
        """GET /me/businesses - for business_management test."""
        
    async def get_waba_info(self, waba_id: str, token: str) -> dict:
        """GET /{waba_id}?fields=id,name,phone_numbers"""
        
    async def get_phone_numbers(self, waba_id: str, token: str) -> list[dict]:
        """GET /{waba_id}/phone_numbers"""
        
    async def get_message_templates(self, waba_id: str, token: str) -> list[dict]:
        """GET /{waba_id}/message_templates"""
```

### AuditService

```python
# backend/src/beautyai_inference/services/audit.py

class AuditService:
    """Audit logging for compliance."""
    
    async def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: int | None = None,
        org_id: int | None = None,
        metadata: dict = {},
        request: Request | None = None
    ) -> AuditLog:
        """Record an audit event."""
```

---

## 7. Implementation Plan

### Phase 1: Security Hardening (MVP - Week 1-2)

**Goal**: Fix critical security issues without breaking existing functionality.

| Task | Effort | Files |
|------|--------|-------|
| Add `MetaCredential` model | 4h | models.py, migration |
| Create `MetaCredentialService` with encryption | 8h | New service |
| Migrate existing `access_token` to vault | 4h | Migration script |
| Add `AuditLog` model + service | 4h | models.py, new service |
| Audit logging middleware | 4h | `api/middleware/` |

### Phase 2: Multi-Tenancy Foundation (Week 3-4)

**Goal**: Enable team collaboration while maintaining backward compatibility.

| Task | Effort | Files |
|------|--------|-------|
| Rename `Customer` → `Organization` | 8h | All files, migration |
| Add `OrganizationMember` model | 4h | models.py |
| Create org membership CRUD endpoints | 8h | New `organizations.py` endpoint |
| Update auth dependencies for org context | 8h | dependencies.py |
| Add `Contact` model | 4h | models.py |

### Phase 3: Meta Integration (Week 5-6)

**Goal**: Proper Meta Business Manager integration.

| Task | Effort | Files |
|------|--------|-------|
| Add `WhatsAppBusinessAccount` model | 4h | models.py |
| Rename `WhatsAppAccount` → `WhatsAppPhoneNumber` | 4h | All files |
| Create `MetaGraphService` | 8h | New service |
| Add `business_management` scope to Embedded Signup | 2h | whatsapp_manager.py |
| Implement token refresh flow | 8h | `MetaCredentialService` |
| Add `MessageTemplate` model | 4h | models.py |

### Phase 4: Long-Term Enhancements (Week 7+)

| Feature | Description |
|---------|-------------|
| Org invites via email | Email-based team invitations |
| RBAC permissions check | Middleware for permission validation |
| Contact tagging/segments | CRM foundation |
| Template sync with Meta | Auto-sync approved templates |
| Webhook subscription management | Outbound webhooks for integrations |
| Quality rating alerts | Notify when Meta quality drops |

---

## 8. Migration Strategy

### Database Migration Order

1. **Add new tables** (non-breaking)
   - `meta_credentials`
   - `audit_logs`
   - `organization_members`
   - `whatsapp_business_accounts`
   - `contacts`

2. **Add columns to existing tables** (non-breaking)
   - `users.is_platform_admin`
   - `customers.meta_business_id`
   - `customers.slug`

3. **Data migration** (offline)
   - Encrypt existing `access_token` values
   - Copy to `meta_credentials`
   - Generate slugs for existing customers

4. **Rename tables** (breaking - requires downtime)
   - `customers` → `organizations`
   - `whatsapp_accounts` → `whatsapp_phone_numbers`

5. **Drop old columns** (cleanup)
   - Remove `whatsapp_accounts.access_token`

---

## Summary

The current architecture works for MVP but needs significant refactoring for production-grade multi-tenant B2B SaaS:

| Area | Status | Priority |
|------|--------|----------|
| Token security | 🔴 Critical | Immediate |
| Audit logging | 🔴 Missing | Week 1 |
| Team collaboration | 🟠 Not possible | Week 2-3 |
| Meta Business Manager | 🟠 Partial | Week 3-4 |
| RBAC | 🟡 Basic only | Week 4+ |
| CRM contacts | 🟡 Not separate | Week 5+ |

**Recommended next step**: Start with Phase 1 (security hardening) to address the critical token storage issue, then proceed with multi-tenancy.





- Remeber what we need to do, and the plan,
- Create new branch on git, regardign to this plan and provide commit message and commit for each step you doing, 
- I confirm your plan, 
- If you agree I can copy the plan and original prompt and start the new session, is it good practice? 


Yes, that's an excellent practice! Starting a fresh session with the plan copied over will give you full context quality.

## Summary to Copy

### Git Branch
```
feature/multi-tenant-architecture-phase1
```

### Plan Summary (copy this to new session)

**Project**: BeautyAI WhatsApp SaaS Platform - Multi-tenant Architecture Refactor

**Phase 1: Security Hardening (MVP - Current Focus)**

| Step | Task | Commit Message |
|------|------|----------------|
| 1 | Add `MetaCredential` model for encrypted token storage | `feat(db): add MetaCredential model for encrypted token vault` |
| 2 | Create `MetaCredentialService` with Fernet encryption | `feat(services): add MetaCredentialService with encryption` |
| 3 | Add `AuditLog` model | `feat(db): add AuditLog model for compliance` |
| 4 | Create `AuditService` | `feat(services): add AuditService for audit logging` |
| 5 | Migration to encrypt existing `access_token` values | `chore(migration): encrypt existing WhatsApp access tokens` |

**Key Files to Create/Modify**:
- models.py - Add `MetaCredential`, `AuditLog`
- `backend/src/beautyai_inference/services/meta_credential.py` - New service
- `backend/src/beautyai_inference/services/audit.py` - New service
- versions - New migration

**Critical Issue Being Fixed**: `WhatsAppAccount.access_token` stored in plaintext → moving to encrypted vault.

---

### Original Prompt (copy this too)

```
You are acting as a senior backend architect.

Context: This is a multi-tenant B2B SaaS platform providing WhatsApp AI bots. 
Each business connects its own WABA via Meta Embedded Signup.

Current task: Implement Phase 1 - Security Hardening
- Add MetaCredential model (encrypted token storage)
- Add AuditLog model
- Create MetaCredentialService with Fernet encryption
- Create AuditService
- Migrate existing plaintext access_tokens

Follow the git workflow: create branch, commit each step separately.
```

---

**Yes, copy the plan + prompt to a new session.** This is the best practice for long tasks - you'll get:
- Fresh 128K token budget
- Full quality responses
- Clean context without noise

