from email.policy import default
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Date, CHAR
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = 'user'

    user_code = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(50), nullable=False)
    email = Column(String(30), nullable=False, unique=True, index=True)
    profile = Column(Integer)
    joinDate = Column(Date, nullable=False)
    clientId = Column(String(255), nullable=False)
    secretId = Column(String(255), nullable=False)
    accessToken = Column(String(255), nullable=False)
    refreshToken = Column(String(255))
    status = Column(String(1), nullable=False, default='N', comment='탈퇴시 Y로 영상 올린 것들때문에')

