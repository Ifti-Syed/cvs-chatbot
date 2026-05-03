"""
User ORM model.

Departments : hr | sales | logistics | production
Roles       : staff | manager | hr_staff | admin
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.auth.database import Base


DEPARTMENTS = {"hr", "sales", "logistics", "production"}
ROLES = {"staff", "manager", "hr_staff", "admin"}


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Access-control fields
    department: Mapped[str] = mapped_column(String(50), nullable=False)
    # "all" is used for admin accounts that span every department
    role: Mapped[str] = mapped_column(String(50), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User {self.email} [{self.department}/{self.role}]>"
