import enum

class UserRole(enum.Enum):
    """User roles for RBAC."""
    USER = "user"      # Regular customer
    ADMIN = "admin"    # Platform administrator (@gmai.sa domain)

try:
    print(f"UserRole('admin'): {UserRole('admin')}")
except Exception as e:
    print(f"UserRole('admin') failed: {e}")

try:
    print(f"UserRole['ADMIN']: {UserRole['ADMIN']}")
except Exception as e:
    print(f"UserRole['ADMIN'] failed: {e}")

try:
    print(f"UserRole['admin']: {UserRole['admin']}")
except Exception as e:
    print(f"UserRole['admin'] failed: {e}")
