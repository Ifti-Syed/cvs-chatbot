"""
Authentication endpoints — Phase 2 (currently disabled in app.py).

POST   /api/auth/login          — email + password → access + refresh tokens
POST   /api/auth/refresh        — refresh token   → new access token
POST   /api/auth/logout         — stateless; client discards tokens
GET    /api/auth/me             — current user profile
POST   /api/auth/users          — admin: create a user
GET    /api/auth/users          — admin / hr: list users
GET    /api/auth/users/{id}     — admin / hr: get a single user
PATCH  /api/auth/users/{id}     — admin: update a user
DELETE /api/auth/users/{id}     — admin: deactivate a user
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.database import get_db
from backend.auth.middleware import (
    UserContext,
    get_current_user,
    require_admin,
    require_hr_admin,
)
from backend.auth.models import User
from backend.auth.schemas import (
    LoginRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserListResponse,
    UserResponse,
    UserUpdate,
)
from backend.auth.service import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user: User | None = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    access  = create_access_token(user.id, user.email, user.department, user.role)
    refresh = create_refresh_token(user.id)
    logger.info("Login: %s (%s/%s)", user.email, user.department, user.role)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    user_id = decode_refresh_token(body.refresh_token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user: User | None = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    access  = create_access_token(user.id, user.email, user.department, user.role)
    refresh = create_refresh_token(user.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(_: UserContext = Depends(get_current_user)):
    return None


@router.get("/me", response_model=UserResponse)
async def me(
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user: User | None = await db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    _: UserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(User).where(User.email == body.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    user = User(
        email=body.email.lower(),
        full_name=body.full_name,
        password_hash=hash_password(body.password),
        department=body.department,
        role=body.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("Created user %s (%s/%s)", user.email, user.department, user.role)
    return user


@router.get("/users", response_model=UserListResponse)
async def list_users(
    _: UserContext = Depends(require_hr_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return UserListResponse(users=list(users), total=len(users))


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: UserContext = Depends(require_hr_admin),
    db: AsyncSession = Depends(get_db),
):
    user: User | None = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    body: UserUpdate,
    _: UserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user: User | None = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.full_name  is not None: user.full_name    = body.full_name
    if body.department is not None: user.department   = body.department
    if body.role       is not None: user.role         = body.role
    if body.is_active  is not None: user.is_active    = body.is_active
    if body.password   is not None: user.password_hash = hash_password(body.password)

    await db.commit()
    await db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: str,
    current_user: UserContext = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account",
        )
    user: User | None = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = False
    await db.commit()
    return None
