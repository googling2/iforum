from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import openai
from dotenv import load_dotenv
import os
from PIL import Image
import urllib.request
import shutil
from moviepy.editor import ImageSequenceClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip, AudioFileClip
import predict
import datetime
from fastapi import Depends, HTTPException
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.orm import Session
from models import User, Fairytale, Voice, Profile, Like
from db import SessionLocal
import json
import uuid
import numpy as np
from sqlalchemy.sql import exists
from starlette.middleware.sessions import SessionMiddleware
import time
import upload
from moviepy.audio.fx.all import audio_fadeout

app = FastAPI()

# SECRET_KEY: 이전에 생성했던 안전한 키 사용
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))

class CustomJinja2Templates(Jinja2Templates):
    def TemplateResponse(self, name: str, context: dict, **kwargs):
        request: Request = context.get("request")
        user_info = request.session.get('user')
        context["user_info"] = user_info
        return super().TemplateResponse(name, context, **kwargs)

templates = CustomJinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_FOLDER2 = "static/myvoice"
if not os.path.exists(UPLOAD_FOLDER2):
    os.makedirs(UPLOAD_FOLDER2)

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
    user_info = request.session.get('user')
    user_code = user_info['usercode'] if user_info else None

    videos = db.query(
        Fairytale.ft_code.label("id"),
        Fairytale.ft_name.label("url"),
        Fairytale.ft_title.label("title"),
        Fairytale.ft_like.label("ft_like"),
        User.user_name.label("name"),
        User.profile.label("img"),
        User.user_code.label("author_id"),
        (db.query(Like).filter(Like.user_code == user_code, Like.ft_code == Fairytale.ft_code).exists()).label("liked")
    ).join(User, Fairytale.user_code == User.user_code).order_by(Fairytale.ft_code.desc()).limit(10).all()

    video_data = [
        {
            "id": video.id,
            "url": video.url if video.url else None,
            "name": video.name if video.name else "",
            "title": video.title if video.title else "",
            "ft_like": video.ft_like,
            "img": f"/static/uploads/{video.img}" if video.img else "/static/uploads/basic.png",
            "liked": video.liked,
            "author_id": video.author_id
        }
        for video in videos
    ]

    return templates.TemplateResponse("index.html", {"request": request, "videos": video_data, "user_code": user_code})

@app.get("/my_profile", response_class=HTMLResponse)
async def my_profile(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    author_id = user_info['usercode']
    return await get_profile(request, db, author_id, user_info)

@app.get("/profile/{author_id}", response_class=HTMLResponse)
async def display_profile(request: Request, author_id: int, db: Session = Depends(get_db)):
    current_user_info = request.session.get('user')
    return await get_profile(request, db, author_id, current_user_info)

async def get_profile(request: Request, db: Session, author_id: int, current_user_info: dict = None):
    profile_user_info = db.query(User).filter(User.user_code == author_id).first()
    if not profile_user_info:
        raise HTTPException(status_code=404, detail="User not found")

    user_fairytales = db.query(Fairytale).filter(Fairytale.user_code == author_id).all()
    total_likes = sum(fairy.ft_like for fairy in user_fairytales)
    user_voices = db.query(Voice).filter(Voice.user_code == author_id).all()
    profile = db.query(Profile).filter(Profile.user_code == author_id).first()
    
    profile_image = f"/static/uploads/{profile.profile_name}" if profile else "/static/uploads/basic.png"

    profile_user_info_dict = {
        "user_code": profile_user_info.user_code,
        "user_name": profile_user_info.user_name,
        "email": profile_user_info.email,
        "profile": profile_user_info.profile,
    }

    print("profile_user_info_dict:", profile_user_info_dict)
    print("current_user_info:", current_user_info)

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "profile_user_info": profile_user_info_dict,
        "current_user_info": current_user_info,
        "fairytales": user_fairytales,
        "voices": user_voices,
        "profile_image": profile_image,
        "total_likes": total_likes,
        "current_user_code": current_user_info['usercode']
    })

@app.get("/upload", response_class=HTMLResponse)
async def display_form(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        user_voices = db.query(Voice).filter(Voice.user_code == user_info['usercode']).all()
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user_voices": user_voices
        })
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/profile", response_class=HTMLResponse)
async def display_profile(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        print("사용자 정보 나오는지 확인:", user_info)
        user_code = user_info['usercode']
        # 동화의 좋아요 수 합산
        user_fairytales = db.query(Fairytale).filter(Fairytale.user_code == user_code).all()
        total_likes = sum(fairy.ft_like for fairy in user_fairytales)
        print("유저 동화 정보:", [f.ft_name for f in user_fairytales])  # 동화 이름을 출력

        user_voices = db.query(Voice).filter(Voice.user_code == user_code).all()
        try:
            profile = db.query(Profile).filter(Profile.user_code == user_code).first()
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
                "voices": user_voices,
                "profile_image": profile_image,
                "total_likes": total_likes,
                "current_user_code": user_code  # 현재 사용자 코드 전달
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
    
    return RedirectResponse(url="/my_profile", status_code=303)

@app.post("/prolike/{ft_code}")
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

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/', status_code=303)

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
        return RedirectResponse(url='/', status_code=303)
    
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
    
    return RedirectResponse(url='/', status_code=303)

@app.get("/gudog", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("gudog.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise


@app.post("/story", response_class=HTMLResponse)
async def create_story(request: Request, keywords: str = Form(...), selected_voice: str = Form(...), selected_mood: str = Form(...), changeImg: str = Form(...), db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        img_dir = "static/img"
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)

        # OpenAI를 사용하여 이야기 생성
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Write in Korean,You are an AI that creates story"},
                {"role": "user", "content": f"{keywords} Using these characters, Title:, write the story in 200 characters including spaces, and divide it into 4 paragraphs."}
            ]
        )

        if completion.choices:
            story_content = completion.choices[0].message.content
            story_title = story_content.split('\n')[0].replace("제목: ", "").replace("Title: ", "").strip()
        else:
            story_content = "텍스트를 다시 입력해주세요!"

        korean_now = datetime.datetime.now() + datetime.timedelta(hours=9)
        timestamp = int(time.time())

        new_story = Fairytale(
            user_code=user_info['usercode'],
            ft_title=story_title,
            ft_name=f"static/final_output{timestamp}.mp4",
            ft_date=korean_now,
            ft_like=0
        )
        db.add(new_story)
        db.commit()

        language = "KR"
        speed = 1.0

        # TTS를 사용하여 오디오 생성
        if selected_voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            audio_response = client.audio.speech.create(
                model="tts-1",
                input=story_content,
                voice=selected_voice
            )
            audio_data = audio_response.content
            audio_file_path = f"static/audio/m1_{uuid.uuid4()}.mp3"
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data)
        else:
            audio_file_path = predict.predict(selected_voice, story_content, language, speed)

        paragraphs = story_content.split('\n\n')
        prompt_paragraphs = [
            f"{changeImg} {paragraph}" for paragraph in paragraphs
        ]
        print("prompt_paragraphs : ",prompt_paragraphs)

        # prompt_paragraphs = [
        #     f"{changeImg} {paragraph}" for paragraph in paragraphs
        # ]
        # print("prompt_paragraphs : ",prompt_paragraphs)
        # # {' '.join(prompt_paragraphs)}
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"""
            {changeImg} please Create a 4-panel image with the same drawing style in a square , The layout is as follows: top left captures the first part, top right captures the second part, and bottom left captures the third part, bottom right captures the fourth part.
            {paragraphs},please {changeImg}
            """,
            size="1024x1024",
            quality="standard",
            n=1
        )

        if response.data:
            image_url = response.data[0].url
            img_filename = f"4cut_image_{uuid.uuid4()}.jpg"
            img_dest = os.path.join("static", "img", img_filename)
            if os.path.exists(img_dest):
                os.remove(img_dest)
            urllib.request.urlretrieve(image_url, img_dest)

            img = Image.open(img_dest)
            crop_sizes = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]

            image_files = []
            for idx, (left, upper, right, lower) in enumerate(crop_sizes):
                cropped_image = img.crop((left, upper, right, lower))
                panel_filename = f"a{idx + 1}_{uuid.uuid4()}.jpg"
                panel_dest = os.path.join("static", "img", panel_filename)
                cropped_image.save(panel_dest)
                image_files.append(panel_dest)


        final_output_file = await create_video(timestamp, selected_mood, audio_file_path, image_files)
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

        return RedirectResponse(
            url=f"/story_view?video_url={final_output_file}&story_title={story_title}&story_content={story_content}",
            status_code=303
        )
    except Exception as e:
        print(f"스토리 생성 및 비디오 생성 중 오류가 발생하였습니다: {e}")
        return HTMLResponse(content=f"스토리 생성 및 비디오 생성 중 오류가 발생하였습니다: {e}", status_code=500)



async def create_video(timestamp, selected_mood, audio_file_path, image_files):
    try:
        def create_image(image_file, output_size=(512, 512)):
            image = Image.open(image_file)
            image = image.resize(output_size, Image.LANCZOS)
            return image

        def create_zoom_in_frames(image, duration=12, fps=24, final_scale=1.3):
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

        def create_zoom_out_frames(image, duration=12, fps=24, initial_scale=1.1):
            num_frames = int(duration * fps)
            zoomed_images = []
            original_center_x, original_center_y = image.width // 2, image.height // 2

            for i in range(num_frames):
                scale = initial_scale - (initial_scale - 1) * (i / num_frames)
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

        def overlay_image_and_audio_on_video(video_file, audio_file, bgm_file, output_file, fadeout_duration=3):
            try:
                video_clip = VideoFileClip(video_file)
                audio_clip = AudioFileClip(audio_file)

                bgm_clip = AudioFileClip(bgm_file).volumex(0.4)
                bgm_duration = video_clip.duration
                bgm_clip = bgm_clip.subclip(0, bgm_duration)
                bgm_clip = bgm_clip.fx(audio_fadeout, fadeout_duration)

                combined_audio = CompositeAudioClip([audio_clip, bgm_clip])

                final_clip = CompositeVideoClip([video_clip.set_audio(combined_audio)])
                final_clip.write_videofile(output_file, codec='libx264')
            except Exception as e:
                print(f"Error overlaying audio on video: {e}")
                raise

        all_zoomed_images = []

        for i, image_file in enumerate(image_files):
            image = create_image(image_file)
            if i % 2 == 0:  # 첫 번째와 세 번째 이미지는 줌인 효과
                zoomed_images = create_zoom_in_frames(image, duration=12, fps=24, final_scale=1.3)
            else:  # 두 번째와 네 번째 이미지는 10% 확대 상태에서 줌아웃 효과
                zoomed_images = create_zoom_out_frames(image, duration=12, fps=24, initial_scale=1.1)
            all_zoomed_images.extend(zoomed_images)

        output_video_file = f'static/output_{uuid.uuid4()}.mp4'
        final_output_file = f'static/final_output{timestamp}.mp4'

        image_to_video(all_zoomed_images, output_video_file, fps=24)

        bgm_file = f'static/bgm/{selected_mood}.mp3'
        overlay_image_and_audio_on_video(output_video_file, audio_file_path, bgm_file, final_output_file)

        # 임시 파일 삭제
        for image_file in image_files:
            os.remove(image_file)
        os.remove(output_video_file)
        os.remove(audio_file_path)

        return final_output_file
    except Exception as e:
        print(f"비디오 생성 중 오류가 발생하였습니다: {e}")
        raise

@app.get("/story_view", response_class=HTMLResponse)
async def story_view(request: Request, video_url: str, story_title: str, story_content: str):
    return templates.TemplateResponse("story.html", {
        "request": request,
        "video_url": video_url,
        "story_title": story_title,
        "story_content": story_content
    })


@app.delete("/delete-video/{ft_code}")
async def delete_video(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code, user_code=user_info['usercode']).first()
    if not fairytale:
        raise HTTPException(status_code=404, detail="Video not found")

    # 좋아요 레코드 삭제
    likes = db.query(Like).filter_by(ft_code=ft_code).all()
    for like in likes:
        db.delete(like)
    
    # 비디오 파일 경로
    video_path = os.path.join('static', fairytale.ft_name)
    
    # 데이터베이스에서 비디오 레코드 삭제
    db.delete(fairytale)
    db.commit()

    # 파일 시스템에서 비디오 파일 삭제
    if os.path.exists(video_path):
        os.remove(video_path)

    return {"message": "비디오 및 관련 좋아요가 삭제되었습니다!"}

@app.post("/like/{ft_code}")
async def like_video(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code).first()

    if not fairytale:
        raise HTTPException(status_code=404, detail="Fairytale not found")

    existing_like = db.query(Like).filter_by(user_code=user_code, ft_code=ft_code).first()

    if existing_like:
        return {"success": True, "message": "Already liked"}

    new_like = Like(user_code=user_code, ft_code=ft_code)
    db.add(new_like)
    fairytale.ft_like = fairytale.ft_like + 1 if fairytale.ft_like is not None else 1
    db.commit()
    return {"success": True, "likes_count": fairytale.ft_like}

@app.post("/unlike/{ft_code}")
async def unlike_video(ft_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    fairytale = db.query(Fairytale).filter_by(ft_code=ft_code).first()

    if not fairytale:
        raise HTTPException(status_code=404, detail="Fairytale not found")

    existing_like = db.query(Like).filter_by(user_code=user_code, ft_code=ft_code).first()

    if not existing_like:
        raise HTTPException(status_code=400, detail="User has not liked this video")

    db.delete(existing_like)
    fairytale.ft_like -= 1
    db.commit()
    return {"success": True, "likes_count": fairytale.ft_like}


@app.get("/check_voice_count")
async def check_voice_count(db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    voice_count = db.query(Voice).filter(Voice.user_code == user_code).count()
    return {"voice_count": voice_count}

@app.post("/upload_voice")
async def upload_voice(file: UploadFile = File(...), voiceName: str = Form(...), db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    voice_count = db.query(Voice).filter(Voice.user_code == user_code).count()
    
    if voice_count >= 1:
        return JSONResponse(content={"message": "최대 1개까지만 업로드할 수 있습니다."}, status_code=400)

    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{os.path.splitext(file.filename)[0]}_{current_date}{file_extension}"
    file_location = os.path.join(UPLOAD_FOLDER2, unique_filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_voice = Voice(
        user_code=user_code,
        voice_name=voiceName,
        voice_date=datetime.datetime.utcnow(),
        voice_status="active",
        voice_filename=unique_filename
    )
    db.add(new_voice)
    db.commit()
    db.refresh(new_voice)

    return {"info": f"파일 '{file.filename}'이 '{unique_filename}'로 업로드되었습니다.", "voiceName": voiceName}

@app.delete("/delete_voice/{voice_code}")
async def delete_voice(voice_code: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    voice = db.query(Voice).filter(Voice.voice_code == voice_code, Voice.user_code == user_code).first()

    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    # 파일 경로
    file_path = os.path.join(UPLOAD_FOLDER2, voice.voice_filename)
    
    # 데이터베이스에서 삭제
    db.delete(voice)
    db.commit()

    # 파일 시스템에서 삭제
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": "목소리가 삭제되었습니다."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
