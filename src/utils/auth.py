"""User authentication and role management for Sentinel Medical AI.

Handles login, password hashing, JWT tokens, and RBAC.
All auth is local — no internet required.
"""

import hashlib
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
import bcrypt
from jose import jwt, JWTError

from src.utils.database import SentinelDB
from src.utils.paths import JWT_SECRET_PATH


# ─── Runtime security flags ──────────────────────────────────────────────────
# Auth is REQUIRED BY DEFAULT. It is switched off only by SENTINEL_DEV_INSECURE=1
# (local development), and SENTINEL_REQUIRE_AUTH=1 forces it back on even then.
# DEV_INSECURE also gates DEV_BYPASS_LICENSE and the OpenAPI docs in server.py.
DEV_INSECURE = os.environ.get("SENTINEL_DEV_INSECURE") == "1"
REQUIRE_AUTH = (not DEV_INSECURE) or os.environ.get("SENTINEL_REQUIRE_AUTH") == "1"


def _load_or_create_secret() -> str:
    """JWT secret: prefer env var, else persist a stable key under DATA_DIR so
    tokens survive restarts and work across workers. (Fix: previously
    regenerated every process start, invalidating all sessions.)"""
    env = os.environ.get("SENTINEL_JWT_SECRET")
    if env:
        return env
    key_file = JWT_SECRET_PATH
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if key_file.exists():
        try:
            return key_file.read_text().strip()
        except Exception:
            pass
    secret = secrets.token_hex(32)
    try:
        key_file.write_text(secret)
        os.chmod(key_file, 0o600)
    except Exception:
        pass
    return secret


# JWT config
SECRET_KEY = _load_or_create_secret()
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 12


def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly."""
    # Truncate to 72 bytes (bcrypt limit)
    pw_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        pw_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(pw_bytes, hashed_bytes)
    except Exception:
        return False


def create_token(user_id: str, username: str, role: str) -> str:
    """Create a JWT token for authenticated sessions."""
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_EXPIRE_HOURS * 3600,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except JWTError:
        return None


# Role → allowed actions. Module-level so endpoints can import it directly.
ROLE_PERMISSIONS = {
    "admin":       {"view", "analyze", "edit_report", "sign_report", "manage_users", "view_audit", "upload"},
    "radiologist": {"view", "analyze", "edit_report", "sign_report", "upload"},
    "technician":  {"upload"},
}


def check_permission(role: str, action: str) -> bool:
    """Module-level RBAC check used by the FastAPI auth dependencies."""
    return action in ROLE_PERMISSIONS.get(role, set())


class AuthManager:
    """Manages user authentication and role-based access control."""

    def __init__(self, db: Optional[SentinelDB] = None):
        self.db = db or SentinelDB()

    def register_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: str = "radiologist",
        created_by: Optional[str] = None,
    ) -> Optional[str]:
        """Register a new user. Returns user ID or None if failed."""
        if role not in ("admin", "radiologist", "technician"):
            logger.error(f"Invalid role: {role}")
            return None
        if not password or len(password) < 6:
            logger.error("Password too short (min 6 chars)")
            return None

        password_hash = hash_password(password)

        try:
            user_id = self.db.create_user(
                username=username,
                password_hash=password_hash,
                full_name=full_name,
                role=role,
            )
            logger.info(f"User registered: {username} ({role})")
            self.db.log_action(created_by or user_id, "register", "user", user_id)
            return user_id
        except Exception as e:
            logger.error(f"Registration failed for {username}: {e}")
            return None

    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change a user's password after verifying the old one."""
        if not new_password or len(new_password) < 6:
            return False
        try:
            conn = self.db._connect()
            row = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
            if row is None:
                conn.close()
                return False
            if not verify_password(old_password, dict(row)["password_hash"]):
                conn.close()
                return False
            conn.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE id=?",
                         (hash_password(new_password), user_id))
            conn.commit()
            conn.close()
            self.db.log_action(user_id, "change_password", "user", user_id)
            return True
        except Exception as e:
            logger.error(f"change_password failed: {e}")
            return False

    def login(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user and return token + user info.

        Returns the API-contract login payload
        {access_token, token_type, user{id, username, full_name, role},
        must_change_password} or None if auth fails.
        """
        import sqlite3

        try:
            conn = self.db._connect()
            row = conn.execute(
                "SELECT * FROM users WHERE username=?", (username,)
            ).fetchone()
            conn.close()
        except Exception as e:
            logger.error(f"Login query failed: {e}")
            return None

        if row is None:
            logger.warning(f"Login failed: user '{username}' not found")
            return None

        user = dict(row)
        if not verify_password(password, user["password_hash"]):
            logger.warning(f"Login failed: wrong password for '{username}'")
            return None

        # Update last login
        conn = self.db._connect()
        conn.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (datetime.now().isoformat(), user["id"]),
        )
        conn.commit()
        conn.close()

        # Generate token
        token = create_token(user["id"], user["username"], user["role"])

        self.db.log_action(user["id"], "login", "user", user["id"])

        logger.info(f"User logged in: {username} ({user['role']})")
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "full_name": user["full_name"],
                "role": user["role"],
            },
            "must_change_password": bool(user.get("must_change_password") or 0),
        }

    def check_permission(self, role: str, action: str) -> bool:
        """Check if a role has permission for an action.

        Role permissions:
        - admin: everything
        - radiologist: view, analyze, edit reports, sign reports
        - technician: upload only
        """
        permissions = {
            "admin": {"view", "analyze", "edit_report", "sign_report", "manage_users", "view_audit", "upload"},
            "radiologist": {"view", "analyze", "edit_report", "sign_report", "upload"},
            "technician": {"upload"},
        }
        return action in permissions.get(role, set())

    def create_default_admin(self):
        """Create default admin account if none exists."""
        conn = self.db._connect()
        admin_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE role='admin'"
        ).fetchone()[0]
        conn.close()

        if admin_count == 0:
            # Default admin password comes from env if set; otherwise a random
            # one-time password is generated and printed ONCE. We never ship a
            # known hardcoded credential. (Fix: was hardcoded 'sentinel2024'.)
            # Either way the account is flagged must_change_password until
            # POST /auth/change_password succeeds.
            pw = os.environ.get("SENTINEL_ADMIN_PASSWORD")
            generated = False
            if not pw:
                pw = secrets.token_urlsafe(12)
                generated = True
            user_id = self.register_user(
                username="admin",
                password=pw,
                full_name="Administrator",
                role="admin",
            )
            if user_id:
                conn = self.db._connect()
                conn.execute("UPDATE users SET must_change_password=1 WHERE id=?", (user_id,))
                conn.commit()
                conn.close()
            if generated:
                banner = [
                    "=" * 70,
                    "  FIRST-RUN ADMIN ACCOUNT CREATED (one-time password)",
                    "  username: admin",
                    f"  password: {pw}",
                    "  ^ Save this now and change it via /auth/change_password.",
                    "  (Set SENTINEL_ADMIN_PASSWORD env to choose your own.)",
                    "=" * 70,
                ]
                # Console (stdout, so a wrapping launcher can capture it) AND the
                # rotating log file in DATA_DIR at WARNING level.
                print("\n".join(banner), flush=True)
                for line in banner:
                    logger.warning(line)
            else:
                logger.warning(
                    "Default admin created with SENTINEL_ADMIN_PASSWORD from env — "
                    "must_change_password is set; change it via /auth/change_password."
                )

    def list_users(self) -> list[dict]:
        """List all users (admin only)."""
        conn = self.db._connect()
        rows = conn.execute(
            "SELECT id, username, full_name, role, created_at, last_login, must_change_password FROM users"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
