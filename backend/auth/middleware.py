"""
JWT authentication middleware / FastAPI dependencies.
"""

from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.auth.service import decode_access_token

_bearer = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    """Lightweight user representation injected into every protected route."""
    id: str
    email: str
    full_name: str
    department: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_hr(self) -> bool:
        return self.role in ("hr_staff", "admin")

    @property
    def is_manager_or_above(self) -> bool:
        return self.role in ("manager", "hr_staff", "admin")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserContext:
    _unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials:
        raise _unauthorized

    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise _unauthorized

    return UserContext(
        id=payload["sub"],
        email=payload["email"],
        full_name=payload.get("full_name", ""),
        department=payload["department"],
        role=payload["role"],
    )


def require_roles(*roles: str) -> Callable:
    async def _check(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role(s): {', '.join(roles)}",
            )
        return user
    return _check


require_admin    = require_roles("admin")
require_hr_admin = require_roles("hr_staff", "admin")
require_manager  = require_roles("manager", "hr_staff", "admin")
