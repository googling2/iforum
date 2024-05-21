from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import openai
from dotenv import load_dotenv
import os
from PIL import Image
import urllib.request
import shutil  # 디렉토리 삭제에 사용
from moviepy.editor import ImageSequenceClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip
from moviepy.editor import AudioFileClip
import predict
import datetime
from fastapi import FastAPI, Depends, Request, HTTPException
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.orm import Session
from models import User, Fairytale, Voice, Base, Profile, Like
from dependencies import get_db
from starlette.status import HTTP_303_SEE_OTHER
from db import SessionLocal
from sqlalchemy.sql import func



import os
from starlette.middleware.sessions import SessionMiddleware
import time
import upload

app = FastAPI()
# SECRET_KEY: 이전에 생성했던 안전한 키 사용
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# 데이터베이스 세션을 반환하는 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request):
    user_info = request.session.get('user')
    if not user_info:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_info

@app.get("/", response_class=HTMLResponse)
async def display_form(request: Request, db: Session = Depends(get_db)):
    # 랜덤으로 필요한 데이터 가져오기
    videos = db.query(
        Fairytale.ft_name.label("url"),
        Fairytale.ft_title.label("title"),
        User.user_name.label("name"),
        Profile.profile_name.label("img")
    ).join(User, Fairytale.user_code == User.user_code)\
     .outerjoin(Profile, User.user_code == Profile.user_code)\
     .order_by(func.random())\
     .limit(10).all()

    video_data = [
        {
            "url": video.url if video.url else None,
            "name": video.name if video.name else "",
            "title": video.title if video.title else "",
            "img": f"/static/uploads/{video.img}" if video.img else "/static/uploads/basic.png"
        }
        for video in videos
    ]

    try:
        return templates.TemplateResponse("index.html", {"request": request, "videos": video_data})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise



# 비디오 업로드 엔드포인트
@app.post("/upload_video")
async def upload_video(request: Request, db: Session = Depends(get_db)):
    return await upload.upload_video(request, db)



@app.get("/upload", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("upload.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/profile", response_class=HTMLResponse)
async def display_profile(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        print("사용자 정보 나오는지 확인:", user_info)
                # 동화의 좋아요 수 합산
        user_fairytales = db.query(Fairytale).filter(Fairytale.user_code == user_info['usercode']).all()

        total_likes = sum(fairy.ft_like for fairy in user_fairytales)
        # 현재 사용자 정보를 이용하여 사용자가 생성한 동화 목록을 가져옴
        user_fairytales = db.query(Fairytale).filter(Fairytale.user_code == user_info['usercode']).all()
        print("유저 동화 정보:", [f.ft_name for f in user_fairytales])  # 동화 이름을 출력

        # Profile 객체 가져오기
        try:
            profile = db.query(Profile).filter(Profile.user_code == user_info['usercode']).first()
            print("Profile 조회 성공:", profile)
        except Exception as e:
            print(f"Profile 조회 중 오류 발생: {e}")
            profile = None
        
        # 프로필 이미지 설정
        if profile:
            print(f"Profile object found: {profile}")
            profile_image = f"/static/uploads/{profile.profile_name}"
        else:
            print("Profile object not found, using default image.")
            profile_image = "/static/uploads/basic.png"
        
        print(f"Profile image 경로 출력: {profile_image}")  # 프로필 이미지 경로를 출력합니다.

        # 템플릿 렌더링
        try:
            response = templates.TemplateResponse("profile.html", {
                "request": request,
                "user_info": user_info,
                "fairytales": user_fairytales,  # 동화 목록을 템플릿에 전달
                "profile_image": profile_image,
                "total_likes": total_likes
            })
            print("템플릿 렌더링 성공")
            return response
        except Exception as e:
            print(f"템플릿 렌더링 중 오류 발생: {e}")
            raise
        
    except Exception as e:
        print(f"전체 코드 실행 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 사용자 정보를 업데이트하는 함수
def update_profile_image(db: Session, user_code: int, file_name: str):
    profile_date = datetime.date.today()
    profile = db.query(Profile).filter(Profile.user_code == user_code).first()
    if profile:
        profile.profile_name = file_name
        profile.profile_date = profile_date
    else:
        profile = Profile(user_code=user_code, profile_name=file_name, profile_date=profile_date)
        db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile

@app.post("/upload_profile_image")
async def upload_profile_image(request: Request, profile_image: UploadFile = File(...), db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    file_location = os.path.join(UPLOAD_FOLDER, profile_image.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(profile_image.file, buffer)
    
    update_profile_image(db, user_info['usercode'], profile_image.filename)
    
    return RedirectResponse(url="/profile", status_code=303)

@app.post("/like/{ft_code}")
async def toggle_like(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user = user_info['usercode']  # 사용자 인증 시스템에서 사용자 ID 가져오기
    existing_like = db.query(Like).filter_by(user_code=user, ft_code=ft_code).first()
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code).first()
    if not fairytale:
        raise HTTPException(status_code=404, detail="Fairytale not found")

    if existing_like:
        db.delete(existing_like)
        fairytale.ft_like -= 1
    else:
        new_like = Like(user_code=user, ft_code=ft_code)
        db.add(new_like)
        fairytale.ft_like += 1
    db.commit()
    return {"likes_count": fairytale.ft_like}


# =================================================================================

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params={'access_type': 'offline', 'prompt': 'consent'},  # access_type과 prompt 추가
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri=os.getenv('GOOGLE_REDIRECT_URI'),
    client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/youtube.readonly'},
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'  # JWKS URI 추가
)



@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri, access_type="offline")


@app.get("/auth")
async def auth(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = dict(token['userinfo'])
    email = user_info['email']
    
    # 데이터베이스에서 해당 이메일을 가진 사용자 검색
    existing_user = db.query(User).filter(User.email == email).first()
    
    if existing_user:
        request.session['user'] = {
            "usercode": existing_user.user_code,
            "name": existing_user.user_name,
            "email": existing_user.email,
            "picture": existing_user.profile
        }
        return RedirectResponse(url='/', status_code=HTTP_303_SEE_OTHER)
    
    # 새 사용자를 데이터베이스에 추가
    new_user = User(
        
        user_name=user_info['name'],
        email=email,
        profile=user_info.get('picture', ''),
        joinDate=datetime.datetime.utcnow(),
        accessToken=token['access_token'],
        refreshToken=token.get('refresh_token', ''),
        status='N'
    )
    db.add(new_user)
    db.commit()

    # 세션에 새 사용자 정보 저장
    request.session['user'] = {
        "usercode": new_user.user_code,
        "name": new_user.user_name,
        "email": new_user.email,
        "picture": new_user.profile
    }
    
    return RedirectResponse(url='/', status_code=HTTP_303_SEE_OTHER)


@app.get("/gudog", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("gudog.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise



@app.post("/story", response_class=HTMLResponse)
async def create_story(request: Request, keywords: str = Form(...), selected_voice: str = Form(...), changeImg: str = Form(...) , db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        # 이미지 디렉토리 초기화
        img_dir = "static/img"
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)  # 디렉토리 삭제
        os.makedirs(img_dir, exist_ok=True)  # 새 디렉토리 생성

        # GPT-3.5로 스토리 생성
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Write in Korean,You are an AI that creates story"},
                {"role": "user", "content": f"{keywords} Using these characters, Title:, write the story in 100 characters including spaces, and divide it into 4 paragraphs."}
            ]
        )

        # 스토리 콘텐츠 확인
        if completion.choices:
            story_content = completion.choices[0].message.content
            story_title = story_content.split('\n')[0].replace("제목: ", "")  # 제목을 첫 줄로 가정
        else:
            story_content = "텍스트를 다시 입력해주세요!"

        korean_now = datetime.datetime.now() + datetime.timedelta(hours=9)  # 현재 한국 시간

        # 타임스탬프 생성
        timestamp = int(time.time())

        # 데이터베이스에 동화 저장
        new_story = Fairytale(
            user_code=user_info['usercode'],
            ft_title=story_title,
            ft_name=f"static/final_output{timestamp}.mp4",  # 비디오 파일 경로를 설정
            ft_date=korean_now,
            ft_like=0
        )
        db.add(new_story)
        db.commit()

        print("이거 나오냐", story_title)
        print("동화내용 나오는지 확인 : ", story_content)

        language = "KR"
        speed = 1.0
        print(selected_voice, "선택한 목소리")

        if selected_voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            # TTS 생성
            audio_response = client.audio.speech.create(
                model="tts-1",
                input=story_content,
                voice=selected_voice
            )
            audio_data = audio_response.content
            audio_file_path = "static/audio/m1.mp3"
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
        else:
            predict.predict(selected_voice, story_content, language, speed)
            audio_file_path = "static/audio/m1.mp3"

        # 이미지 생성 및 저장
        image_paths = []
        paragraphs = story_content.split('\n\n')
        delay_seconds = 15

        response = client.images.generate(
            model="dall-e-3",
            prompt=f"""
            "Create a 4-panel image with the same drawing style in a square {changeImg} The layout is as follows: top left captures the first part, top right captures the second part, and bottom left captures the third part. , the fourth section appears in the lower right corner."
            {paragraphs} 
            """,
            size="1024x1024",
            quality="standard",
            n=1
        )

        if response.data:
            image_url = response.data[0].url
            img_filename = "4cut_image.jpg"
            img_dest = os.path.join("static", "img", img_filename)
            if os.path.exists(img_dest):
                os.remove(img_dest)
            urllib.request.urlretrieve(image_url, img_dest)

            # 이미지 열기
            img = Image.open(img_dest)

            crop_sizes = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]

            # 자른 이미지 저장
            for idx, (left, upper, right, lower) in enumerate(crop_sizes):
                # 부분 이미지 추출
                cropped_image = img.crop((left, upper, right, lower))
                # 저장할 파일명 설정
                panel_filename = f"a{idx + 1}.jpg"
                panel_dest = os.path.join("static", "img", panel_filename)
                # 파일이 이미 존재하면 삭제
                if os.path.exists(panel_dest):
                    os.remove(panel_dest)
                # 부분 이미지 저장
                cropped_image.save(panel_dest)

        # 비디오 생성 호출
        final_output_file = await create_video(timestamp)

        print("video_url 출력해보기 ", final_output_file)
        

        # 리디렉션
        return RedirectResponse(
            url=f"/story_view?video_url={final_output_file}&story_title={story_title}&story_content={story_content}",
            status_code=HTTP_303_SEE_OTHER
        )
    except Exception as e:
        print(f"스토리 생성 및 비디오 생성 중 오류가 발생하였습니다: {e}")
        return HTMLResponse(content=f"스토리 생성 및 비디오 생성 중 오류가 발생하였습니다: {e}", status_code=500)

import json


@app.get("/story_view", response_class=HTMLResponse)
async def story_view(request: Request, video_url: str, story_title: str, story_content: str):

   
    with open('video_url.json', 'w') as f:
     json.dump({"video_url": video_url,"story_title": story_title}, f)

    return templates.TemplateResponse("story.html", {
        "request": request,
        "video_url": video_url,
        "story_title": story_title,
        "story_content": story_content
    })

async def create_video(timestamp):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from moviepy.editor import ImageSequenceClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip
        import os

        def create_image(image_file, output_size=(512, 512)):
            image = Image.open(image_file)
            image = image.resize(output_size, Image.LANCZOS)
            return image

        def create_zoom_frames(image, duration=6, fps=24, final_scale=1.3):
            num_frames = int(duration * fps)
            zoomed_images = []
            original_center_x, original_center_y = image.width // 2, image.height // 2

            for i in range(num_frames):
                scale = 1 + (final_scale - 1) * (i / num_frames)
                new_width = int(image.width * scale)
                new_height = int(image.height * scale)

                if new_width % 2 != 0:
                    new_width += 1
                if new_height % 2 != 0:
                    new_height += 1

                frame = image.resize((new_width, new_height), Image.LANCZOS)
                new_center_x, new_center_y = frame.width // 2, frame.height // 2
                left = max(0, new_center_x - original_center_x)
                top = max(0, new_center_y - original_center_y)
                right = left + image.width
                bottom = top + image.height
                frame = frame.crop((left, top, right, bottom))
                zoomed_images.append(np.array(frame))

            return zoomed_images

        def image_to_video(images, output_file, fps=24):
            try:
                clip = ImageSequenceClip(images, fps=fps)
                clip.write_videofile(output_file, codec='libx264')
            except Exception as e:
                print(f"Error creating video: {e}")
                raise

        def overlay_image_and_audio_on_video(video_file, audio_file, output_file):
            try:
                video_clip = VideoFileClip(video_file)
                audio_clip = AudioFileClip(audio_file)
                final_clip = CompositeVideoClip([video_clip.set_audio(audio_clip)])
                final_clip.write_videofile(output_file, codec='libx264')
            except Exception as e:
                print(f"Error overlaying audio on video: {e}")
                raise

        base_path = 'static/img'
        image_files = []
        idx = 1
        while True:
            file_path = os.path.join(base_path, f'a{idx}.jpg')
            if os.path.exists(file_path):
                image_files.append(file_path)
                idx += 1
            else:
                break

        if not image_files:
            print("No image files found.")
            return

        audio_clip = AudioFileClip('static/audio/m1.mp3')
        total_duration = audio_clip.duration
        duration_per_image = total_duration / len(image_files)
        all_zoomed_images = []

        for image_file in image_files:
            image = create_image(image_file)
            zoomed_images = create_zoom_frames(image, duration=duration_per_image, fps=24, final_scale=1.3)
            all_zoomed_images.extend(zoomed_images)

        output_video_file = f'static/output{timestamp}.mp4'
        final_output_file = f'static/final_output{timestamp}.mp4'

        image_to_video(all_zoomed_images, output_video_file, fps=24)
        overlay_image_and_audio_on_video(output_video_file, 'static/audio/m1.mp3', final_output_file)

        return final_output_file
    except Exception as e:
        print(f"비디오 생성 중 오류가 발생하였습니다: {e}")
        raise

@app.post("/like/{ft_code}")
async def toggle_like(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']  # 사용자 코드를 세션에서 가져옴
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code).first()

    if not fairytale:
        raise HTTPException(status_code=404, detail="Fairytale not found")

    existing_like = db.query(Like).filter_by(user_code=user_code, ft_code=ft_code).first()

    if existing_like:
        db.delete(existing_like)
        fairytale.ft_like = (fairytale.ft_like - 1 if fairytale.ft_like > 0 else 0)
    else:
        new_like = Like(user_code=user_code, ft_code=ft_code)
        db.add(new_like)
        fairytale.ft_like = (fairytale.ft_like + 1 if fairytale.ft_like is not None else 1)
    
    db.commit()
    return {"likes_count": fairytale.ft_like}

@app.delete("/delete-video/{ft_code}")
async def delete_video(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code, user_code=user_info['usercode']).first()
    if not fairytale:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join('static', fairytale.ft_name)
    db.delete(fairytale)
    db.commit()

    if os.path.exists(video_path):
        os.remove(video_path)

    return {"message": "비디오 삭제가 완료되었습니다!"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)