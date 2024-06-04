import json
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
import os
from dependencies import get_db
from models import Fairytale, User
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import random
import time
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from argparse import Namespace
from requests.exceptions import ConnectionError, Timeout
from google.auth.transport.requests import Request as GoogleRequest

# app = FastAPI()
# app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]
RETRIABLE_EXCEPTIONS = (HttpError, ConnectionError, Timeout)
MAX_RETRIES = 5

# @app.post("/upload_video")
# async def upload_video(request: Request, db: Session = Depends(get_db)):
async def upload_video(request, db):
    print("video_url:")
    user = request.session.get("user")
    usercode = user.get("usercode") if user else None
    if not usercode:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="User not logged in")

    user_access_token = get_user_access_token(db, usercode)
    print(user_access_token)
    user_refresh_token = get_user_refresh_token(db, usercode)
    print(user_refresh_token)
    if not user_access_token or not user_refresh_token:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid user credentials")
    print("video_url:")
    user_credentials = Credentials(
        token=user_access_token,
        refresh_token=user_refresh_token,
        token_uri='https://accounts.google.com/o/oauth2/token',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        scopes=['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.force-ssl']
    )

    print("video_url:")
    with open('video_url.json', 'r') as f:
        data = json.load(f)

    video_url = data['video_url']
    story_title = data['story_title']
    print("video_url:",video_url)


    ft_title = get_ft_title_by_video_url(db, video_url)
    if not ft_title:
            raise HTTPException(status_code=404, detail="Fairytale not found")

    try:
        youtube = get_authenticated_service(user_credentials, db, usercode)

        options = Namespace(
            file=video_url,
            title=f"[IFORUM] AI창작 부업 - {ft_title}",
            description="AI 동화 만들기\n누구나 쉽게 AI동화를 창작합니다.\n유튜브에 직접 업로드 하지 않고 자동으로 업로드가 됩니다!\n\n기발한 아이디어? 편집? 직접 업로드?\n이제 단 한 줄을 작성하면 이 모든 것을 손쉽게 할 수 있습니다!\nAI 부업 시대, 여러분을 기다립니다.\n www.iforum.shop",
            category="24",
            keywords="side hustle,fairytale,부업,side job,동화,어린이,교육,창작,이야기,동화책,판타지",
            privacyStatus="public"
        )

        initialize_upload(youtube, options)
        return RedirectResponse(url="/my_profile", status_code=302)
    except HttpError as e:
        if e.resp.status == 400 and "uploadLimitExceeded" in e.content.decode():
            return JSONResponse(status_code=400, content={"message": "The user has exceeded the number of videos they may upload."})
        else:
            return JSONResponse(status_code=500, content={"message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

def refresh_access_token(credentials: Credentials):
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleRequest())
    return credentials

def get_authenticated_service(credentials: Credentials, db: Session, usercode: int):
    try:
        credentials = refresh_access_token(credentials)

        user = db.query(User).filter(User.user_code == usercode).first()
        if user:
            user.accessToken = credentials.token
            user.refreshToken = credentials.refresh_token
            db.commit()

        return build('youtube', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Error creating YouTube service: {e}")
        raise

def get_ft_title_by_video_url(db: Session, video_url: str):
    print(f"Looking for Fairytale with ft_name: {video_url}")
    fairytale = db.query(Fairytale).filter(Fairytale.ft_name == video_url).first()
    if fairytale:
        print(f"Found Fairytale: {fairytale.ft_title}")
        return fairytale.ft_title
    else:
        print("Fairytale not found")
        print(f"Found Fairytale: {fairytale.ft_title}")
        return None

def get_user_access_token(db: Session, usercode: int):
    user = db.query(User).filter(User.user_code == usercode).first()
    return user.accessToken if user else None

def get_user_refresh_token(db: Session, usercode: int):
    user = db.query(User).filter(User.user_code == usercode).first()
    return user.refreshToken if user else None

def initialize_upload(youtube, options):
    body = {
        "snippet": {
            "title": options.title,
            "description": options.description,
            "tags": options.keywords.split(","),
            "categoryId": options.category
        },
        "status": {
            "privacyStatus": options.privacyStatus,
            "madeForKids": True,
            "selfDeclaredMadeForKids": True  
        }
    }

    media_body = MediaFileUpload(options.file, chunksize=-1, resumable=True)
    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media_body
    )

    resumable_upload(insert_request)

def resumable_upload(insert_request):
    response = None
    error = None
    retry = 0
    print("insert++++",type(insert_request))
    while response is None:
        try:
            print("Uploading file...")
            status, response = insert_request.next_chunk()
            print("Status:", status)
            print("Response:", response)
            if response is not None and 'id' in response:
                print(f"Video id '{response['id']}' was successfully uploaded.")
                return
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                print("Non-retriable HTTP error:", e)
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable error occurred: {e}"
            print(error)

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                raise Exception("No longer attempting to retry.")
            sleep_seconds = random.random() * (2 ** retry)
            print(f"Sleeping {sleep_seconds} seconds and then retrying...")
            time.sleep(sleep_seconds)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)