from email.policy import default
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Date, CHAR
from sqlalchemy.orm import relationship
from db import Base

class User(Base):
    __tablename__ = "user"

    user_code = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(50), nullable=False)
    email = Column(String(30), nullable=False)
    profile = Column(Integer)
    joinDate = Column(Date, nullable=False)
    clientId = Column(String(255), nullable=False)
    secretId = Column(String(255), nullable=False)
    accessToken = Column(String(255), nullable=False)
    refreshToken = Column(String(255), nullable=False)
    status = Column(CHAR(1), default='N', nullable=False)

    voice_records = relationship("VoiceRecord", back_populates="user")

class VoiceRecord(Base):
    __tablename__ = "voice"

    voice_code = Column(Integer, primary_key=True, autoincrement=True)
    user_code = Column(Integer, ForeignKey('user.user_code'), nullable=False)
    voice_name = Column(String(255), nullable=False)
    voice_date = Column(Date, nullable=False)
    voice_status = Column(CHAR(1), default='N', nullable=False)

    user = relationship("User", back_populates="voice")