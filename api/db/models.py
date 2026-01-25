import uuid

from datetime import datetime

from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.dialects.postgresql import UUID

from api.db.db_config import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)