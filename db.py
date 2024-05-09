from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
load_dotenv()  # 환경변수 로드
# 환경변수에서 데이터베이스 설정 정보를 가져옴
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL)
# 세션 생성자 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 베이스 모델 생성
Base = declarative_base()