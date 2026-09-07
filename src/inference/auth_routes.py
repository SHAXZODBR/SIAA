"""Authentication API routes + reusable auth dependencies for Sentinel."""

import os
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from src.utils.auth import (
    AuthManager, verify_token, check_permission, REQUIRE_AUTH, DEV_INSECURE,
)
from src.utils.database import SentinelDB

# Auth is REQUIRED BY DEFAULT (REQUIRE_AUTH=True). Only SENTINEL_DEV_INSECURE=1
# relaxes it for local development; SENTINEL_REQUIRE_AUTH=1 forces it back on.
# The flags live in src.utils.auth so server.py can gate docs/license on them.

router = APIRouter(prefix="/auth", tags=["authentication"])

# Shared instances
db = SentinelDB()
auth = AuthManager(db)

# Create default admin on import (logs a loud warning to force a password change)
auth.create_default_admin()

_bearer = HTTPBearer(auto_error=False)


# ─── Reusable auth dependencies (import these into server.py endpoints) ──────
def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> dict:
    """Validate the Bearer token and return the user payload. 401 if invalid."""
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    payload = verify_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def require_permission(action: str):
    """Dependency factory: enforce that the user's role permits `action`."""
    def _dep(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "")
        if not check_permission(role, action):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' is not permitted to '{action}'",
            )
        return user
    return _dep


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


def clinical_auth(action: str = "analyze"):
    """Endpoint guard for clinical routes. Enforces auth+RBAC by default;
    permissive (returns an anonymous admin principal) ONLY when
    SENTINEL_DEV_INSECURE=1 and SENTINEL_REQUIRE_AUTH is not forcing it on.
    Returns the user dict either way."""
    def _dep(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> dict:
        if not REQUIRE_AUTH and DEV_INSECURE:
            # Dev-insecure mode — allow, but tag the principal so audit logs show it.
            if creds and creds.credentials:
                payload = verify_token(creds.credentials)
                if payload:
                    return payload
            return {"sub": "anonymous-dev", "username": "anonymous", "role": "admin"}
        # Default — strict.
        if creds is None or not creds.credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
        payload = verify_token(creds.credentials)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        if not check_permission(payload.get("role", ""), action):
            raise HTTPException(status_code=403,
                                detail=f"Role '{payload.get('role')}' cannot '{action}'")
        return payload
    return _dep


# ─── Request models ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    fullName: str
    role: str = "radiologist"


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class LoginUser(BaseModel):
    id: str
    username: str
    full_name: str
    role: str


class LoginResponse(BaseModel):
    """API contract v1: {access_token, token_type, user, must_change_password}."""
    access_token: str
    token_type: str = "bearer"
    user: LoginUser
    must_change_password: bool = False


# ─── Routes ──────────────────────────────────────────────────────────────────
@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request):
    """Authenticate user and return a JWT bearer token."""
    result = auth.login(req.username, req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return result


@router.post("/register")
async def register(req: RegisterRequest, admin: dict = Depends(require_admin)):
    """Register a new user — ADMIN ONLY (fix: was anonymous + could self-grant admin)."""
    # Only an admin may create an admin account.
    if req.role == "admin" and admin.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create admin users")
    user_id = auth.register_user(
        username=req.username,
        password=req.password,
        full_name=req.fullName,
        role=req.role,
        created_by=admin.get("sub"),
    )
    if user_id is None:
        raise HTTPException(status_code=400, detail="Registration failed (username taken?)")
    return {"userId": user_id, "message": "User registered successfully"}


@router.post("/change_password")
async def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """Change your own password (required after first login for the default admin)."""
    ok = auth.change_password(user.get("sub"), req.old_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Old password incorrect or new password too weak")
    return {"message": "Password changed"}


@router.get("/me")
async def whoami(user: dict = Depends(get_current_user)):
    """Return the current authenticated user."""
    row = db.get_user(user.get("sub")) or {}
    return {
        "id": user.get("sub"),
        "username": user.get("username"),
        "full_name": row.get("full_name"),
        "role": user.get("role"),
        "must_change_password": bool(row.get("must_change_password") or 0),
    }


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    """List all users — ADMIN ONLY."""
    return auth.list_users()


@router.get("/stats")
async def get_stats(admin: dict = Depends(require_admin)):
    """Database statistics — ADMIN ONLY."""
    return db.get_stats()
