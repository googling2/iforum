from email.policy import default
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Date, CHAR
from sqlalchemy.orm import relationship
from db import Base
import datetime


class Profile(Base):
    __tablename__ = 'profile'

    profile_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, ForeignKey('user.user_code'), nullable=False)
    profile_date = Column(Date, nullable=False, default=datetime.date.today)
    profile_name = Column(String(255), nullable=False)

    user = relationship("User", back_populates="profiles")   


class Like(Base):
    __tablename__ = 'likes'

    like_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, ForeignKey('user.user_code'), nullable=False)
    ft_code = Column(Integer, ForeignKey('fairytale.ft_code'), nullable=False)

    user = relationship("User", back_populates="likes")
    fairytale = relationship("Fairytale", back_populates="likes")

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

    fairytales = relationship("Fairytale", back_populates="user")
    voices = relationship("Voice", back_populates="user")
    profiles = relationship("Profile", back_populates="user")
    likes = relationship("Like", back_populates="user")



class Fairytale(Base):
    __tablename__ = 'fairytale'

    ft_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, ForeignKey('user.user_code'))
    ft_title = Column(String(255), nullable=False)
    ft_date = Column(Date, nullable=False)
    ft_name = Column(String(255), nullable=False)
    ft_like = Column(Integer)
    ft_youtubeLink = Column(String(255))

    user = relationship("User", back_populates="fairytales")
    likes = relationship("Like", back_populates="fairytale")


class Voice(Base):
    __tablename__ = 'voice'

    voice_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, ForeignKey('user.user_code'), nullable=False)
    voice_name = Column(String(255), nullable=False)
    voice_date = Column(Date, nullable=False)
    voice_status = Column(CHAR(1), nullable=False, default='N', comment='Y일시 남들이 사용가능')
    voice_filename = Column(String(255))

    # 관계 정의
    user = relationship("User", back_populates="voices")


    # 관계 정의

User.fairytales = relationship("Fairytale", order_by=Fairytale.ft_code, back_populates="user")