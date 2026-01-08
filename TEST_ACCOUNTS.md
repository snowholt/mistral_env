# BeautyAI SaaS Platform - Test Accounts

## 🎉 Setup Complete!

### ✅ Chatbot Widget
The AI assistant chatbot is now live on the public website pages (landing page, privacy, terms). It's configured in **demo mode** with Arabic language support.

- Widget appears on all public pages (not in dashboard)
- Demo responses in Arabic
- Located at bottom-right corner
- Opens on click with chat interface

### 📋 Test Accounts

#### ADMIN ACCOUNT
For testing the Admin Dashboard:
```
Email:    nariman@gmai.sa
Password: Admin@123456
Role:     ADMIN
```

**Login URL:** https://portal.gmai.sa/login

**Admin Dashboard:**
- Customers: https://portal.gmai.sa/app/admin/customers
- Metrics: https://portal.gmai.sa/app/admin/metrics
- Users: https://portal.gmai.sa/app/admin/users

---

#### CUSTOMER ACCOUNT
For testing the Customer Dashboard:
```
Email:    customer@test.com
Password: Customer@123456
Role:     USER
```

**Login URL:** https://portal.gmai.sa/login

**Customer Dashboard:** https://portal.gmai.sa/app

---

## 🚀 How to Create Accounts

The accounts were created using:
```bash
cd /home/lumi/beautyai/backend
source venv/bin/activate
python scripts/simple_create_users.py
```

## 🔧 Manual Account Creation

If you need to create accounts manually, you can use psql:

```sql
-- Admin account
INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
VALUES ('nariman@gmai.sa', '<bcrypt_hash>', 'Nariman Admin', 'admin', true, true);

-- Customer account  
INSERT INTO users (email, password_hash, full_name, role, is_active, is_verified)
VALUES ('customer@test.com', '<bcrypt_hash>', 'Test Customer', 'user', true, true);
```

To generate a password hash:
```python
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
hash = pwd_context.hash("YourPasswordHere")
print(hash)
```

---

## 🧪 Testing Checklist

### Public Website (https://portal.gmai.sa)
- [ ] Homepage loads
- [ ] AI chatbot button appears bottom-right
- [ ] Chatbot opens and shows Arabic welcome message
- [ ] Can send messages and get demo responses
- [ ] Login/Register pages accessible

### Admin Dashboard
- [ ] Login with nariman@gmai.sa
- [ ] Access admin routes (/app/admin/*)
- [ ] View customers list
- [ ] View metrics
- [ ] View users list
- [ ] Regular users cannot access admin routes

### Customer Dashboard
- [ ] Login with customer@test.com
- [ ] Access customer dashboard (/app)
- [ ] Cannot access admin routes (403/redirect)
- [ ] Dashboard shows user info

---

## 📝 Notes

1. **Email Verification Skipped**: Test accounts have `is_verified=true` to skip email verification
2. **Database**: PostgreSQL database `beautyai` on localhost
3. **Password Security**: Uses bcrypt with 12 rounds
4. **Frontend Build**: Located at `_website_snapshot/gmai.sa/gmai.sa/dist/`
5. **Nginx Config**: `/etc/nginx/sites-enabled/portal.gmai.sa`

---

## 🔄 Rebuild Frontend

If you make changes to the ChatWidget or any frontend code:

```bash
cd /home/lumi/beautyai/_website_snapshot/gmai.sa/gmai.sa
./node_modules/.bin/vite build
```

Then reload nginx:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

**Created:** December 22, 2025  
**Status:** ✅ Ready for testing
