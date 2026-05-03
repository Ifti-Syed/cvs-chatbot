"""
Pydantic schemas for authentication and user management.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from backend.auth.models import DEPARTMENTS, ROLES


# ── Request bodies ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserCreate(BaseModel):
    """Admin-only: create a new user."""
    email: str
    full_name: str
    password: str
    department: str
    role: str

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        valid = DEPARTMENTS | {"all"}
        if v not in valid:
            raise ValueError(f"department must be one of: {sorted(valid)}")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"role must be one of: {sorted(ROLES)}")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class UserUpdate(BaseModel):
    """Admin-only: update an existing user."""
    full_name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid = DEPARTMENTS | {"all"}
        if v not in valid:
            raise ValueError(f"department must be one of: {sorted(valid)}")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in ROLES:
            raise ValueError(f"role must be one of: {sorted(ROLES)}")
        return v


# ── Response bodies ───────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    department: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
