from email.policy import default
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Date, CHAR
from sqlalchemy.orm import relationship
from db import Base

class User(Base):
    __tablename__ = 'user'

    user_code = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(50), nullable=False)
    email = Column(String(30), nullable=False, unique=True, index=True)
    profile = Column(Integer)
    joinDate = Column(Date, nullable=False)
    accessToken = Column(String(255), nullable=False)
    refreshToken = Column(String(255))
    status = Column(String(1), nullable=False, default='N', comment='탈퇴시 Y로 영상 올린 것들때문에')

class Fairytale(Base):
    __tablename__ = 'fairytale'

    ft_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, ForeignKey('user.user_code'))
    ft_title = Column(String(255), nullable=False)
    ft_date = Column(Date, nullable=False)
    ft_name = Column(String(255), nullable=False)
    ft_like = Column(Integer)
    ft_youtubeLink = Column(String(255))

    # 관계 정의
    user = relationship("User", back_populates="fairytales")

User.fairytales = relationship("Fairytale", order_by=Fairytale.ft_code, back_populates="user")