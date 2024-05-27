from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 데이터베이스 모델 정의
class Fairytale(Base):
    __tablename__ = 'fairytale'

    ft_code = Column(Integer, primary_key=True, index=True)
    user_code = Column(Integer, index=True)
    ft_title = Column(String, index=True)
    ft_date = Column(DateTime)
    ft_name = Column(String)
    ft_like = Column(Integer)
    ft_youtubeLink = Column(String)

# FastAPI 앱 초기화
app = FastAPI()

# 데이터베이스 세션을 반환하는 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 로컬 디렉토리와 데이터베이스를 비교하여 데이터베이스에 없는 파일 삭제 (DELETE)
@app.delete("/cleanup_files")
async def cleanup_files(db: Session = Depends(get_db)):
    upload_folder = "static"  # 파일이 저장된 로컬 디렉토리 경로

    # 데이터베이스에 저장된 파일 목록 가져오기
    db_files = db.query(Fairytale.ft_name).all()
    db_files = set(file.ft_name for file in db_files)

    print(f"DB 파일 목록: {db_files}")

    # 로컬 디렉토리에 저장된 파일 목록 가져오기 (하위 폴더 제외)
    local_files = set()
    for file in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, file).replace("\\", "/")
        if os.path.isfile(file_path):
            local_files.add(file_path)

    print(f"로컬 파일 목록: {local_files}")

    # 데이터베이스에 없는 파일 목록 찾기
    files_to_delete = local_files - db_files

    print(f"삭제할 파일 목록: {files_to_delete}")

    # 파일 삭제
    deleted_files = []
    for file in files_to_delete:
        if os.path.isfile(file):
            os.remove(file)
            deleted_files.append(file)
            print(f"삭제된 파일: {file}")

    return {"deleted_files": deleted_files}

# 로컬 디렉토리와 데이터베이스를 비교하여 데이터베이스에 없는 파일 목록 확인 (GET)
@app.get("/cleanup_files_check")
async def cleanup_files_check(db: Session = Depends(get_db)):
    upload_folder = "static"  # 파일이 저장된 로컬 디렉토리 경로

    # 데이터베이스에 저장된 파일 목록 가져오기
    db_files = db.query(Fairytale.ft_name).all()
    db_files = set(file.ft_name for file in db_files)

    print(f"DB 파일 목록: {db_files}")

    # 로컬 디렉토리에 저장된 파일 목록 가져오기 (하위 폴더 제외)
    local_files = set()
    for file in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, file).replace("\\", "/")
        if os.path.isfile(file_path):
            local_files.add(file_path)

    print(f"로컬 파일 목록: {local_files}")

    # 데이터베이스에 없는 파일 목록 찾기
    files_to_delete = local_files - db_files

    print(f"삭제할 파일 목록: {files_to_delete}")

    return {"files_to_delete": list(files_to_delete)}

if __name__ == "__main__":
    import uvicorn
    # 데이터베이스 테이블 생성
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host='127.0.0.1', port=8000)





# 실행을 하고
# http://127.0.0.1:8000/cleanup_files_check 주소창에 입력
# 그다음에 터미널 아무거나 열고
# curl -X DELETE http://127.0.0.1:8000/cleanup_files
# 명령어 실행

