## Clarifying Questions for "Request Demo" Feature

### 1. Demo Request Form Fields & Flow
**Why asking**: Need to determine what information to collect and validation rules.

- **Q1.1**: Should we use the existing ContactForm component with its current fields 
  (firstName, lastName, email, phone, company, companySize, message)?
  - Option A: Yes, reuse it as-is


- **Q1.2**: Should users fill out the form BEFORE or AFTER guest registration?
  - Option A: Form → Submit → Auto-create guest account → Admin approval → Grant access → Email with registration link

### 2. Guest User System
**Why asking**: Need to define guest account behavior and limitations.

- **Q2.1**: How should guest accounts differ from regular user accounts?
  - Option A: Separate `GuestUser` table (simpler, isolated)


- **Q2.2**: What should happen after guest registration?
  - Option C: Guest account stays dormant until admin grants access, and then Guest receives email notification

- **Q2.3**: Should guests have a customer dashboard or just access to the demo?
  - You mentioned: "we will show it on the customer dashboard"
  - Option B: Guest sees full customer dashboard but with demo only (no agent setup, etc.), all other parts will be gray/disabled.

### 3. Demo Access Control & Usage Limits
**Why asking**: Need to implement proper security and prevent abuse.

- **Q3.1**: How should demo access be limited?
  - Option C: Both time + usage limits
  - Option D: Admin manually disables when needed (no auto-expiry)

- **Q3.2**: What happens when limits are reached?
  - Option A: Demo interface shows "Demo expired - Contact us to upgrade"


- **Q3.3**: Should the demo interface show usage metrics to the guest?
  - Example: "You have 5/10 demo conversations remaining"
  - Option A: Yes, show usage counter
  - Option C: Show countdown for time-based limits also

### 4. Admin Dashboard Integration
**Why asking**: Need to design the admin workflow and UI placement.

- **Q4.1**: Where should the "Demo Requests" page appear in admin navigation?
  - Current menu: Customers, Users, Metrics
  - Option B: Add under Customers as sub-menu


- **Q4.2**: What actions should admins be able to take on demo requests?
  - Option C: Approve/Reject + Set custom limits + Schedule follow-up

- **Q4.3**: Should admins be notified of new demo requests?
  - Option B: Email to specific admin (configurable)


### 5. Demo Interface Integration
**Why asking**: Need to embed test_lean.html properly in customer/guest dashboard.

- **Q5.1**: How should test_lean.html be integrated into the React dashboard?
  - Option B: Rewrite as React component (cleaner, more maintainable)

- **Q5.2**: Should the demo interface be customizable per guest?
  - Example: Pre-select language, custom greeting, etc.
  - Option B: No, all guests see same interface

### 6. Upgrade Path
**Why asking**: Need to plan conversion from guest to paid customer.

- **Q6.1**: How should guests convert to paying customers?
  - Option A: "Upgrade" button in demo → Standard registration flow → Payment (just add it not develop it now)


- **Q6.2**: Should demo conversation history be preserved after upgrade?
  - Option B: No, fresh start


### 7. Website Frontend Changes
**Why asking**: Need to determine CTA placement and visibility.

- **Q7.1**: Where should "Request Demo" button appear on the website?
  - Option A: Hero section CTA (replace/add to existing CTA)
  - Option B: Navigation bar (header)
  - Option C: Both + Footer

- **Q7.2**: Should existing "Contact Us" form still exist separately?
  - Option B: Keep both (Contact Us for general inquiries, Request Demo for trials)
