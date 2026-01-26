import uuid

from sqlalchemy import Column, BigInteger, String, DateTime, ForeignKey, func, Enum, Float
from sqlalchemy.dialects.postgresql import UUID

from api.db.db_config import Base
from api.db.schemas import TaskType


class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_telegram_id = Column(BigInteger, ForeignKey('users.telegram_id'), nullable=False, index=True)
    task_type = Column(Enum(TaskType), nullable=False)
    best_model = Column(String, nullable=False)
    target = Column(String, nullable=False)
    metric = Column(Float, nullable=True)
    dataset_hash = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())