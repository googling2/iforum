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
# import predict
import datetime
from fastapi import Depends, HTTPException
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.orm import Session
from models import User, Fairytale, Voice, Profile, Like, Subscribe
from db import SessionLocal
import json
import uuid
import predict
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


@app.post("/search", response_class=HTMLResponse)
async def search_fairytales(request: Request, keyword: str = Form(...), db: Session = Depends(get_db)):
    user_info = request.session.get('user')
    user_code = user_info['usercode'] if user_info else None

    results = db.query(Fairytale).filter(Fairytale.ft_title.ilike(f'%{keyword}%')).all()
    video_data = [
        {
            "id": result.ft_code,
            "url": result.ft_name if result.ft_name else None,
            "title": result.ft_title if result.ft_title else "",
            "ft_like": result.ft_like,
            "img": f"/static/uploads/{result.user.profile}" if result.user.profile else "/static/uploads/basic.png",
            "name": result.user.user_name if result.user.user_name else "",
        }
        for result in results
        
    ]

    profile_user_info, profile_image, follow_count, follower_count, total_likes = (None, "/static/uploads/basic.png", 0, 0, 0)
    if user_info:
        profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_code)

    return templates.TemplateResponse("main.html", {
        "request": request,
        "videos": video_data,
        "profile_user_info": profile_user_info,
        "profile_image": profile_image,
        "follow_count": follow_count,
        "follower_count": follower_count,
        "total_likes": total_likes
    })


@app.get("/main", response_class=HTMLResponse)
async def display_form(request: Request, db: Session = Depends(get_db)):
    user_info = request.session.get('user')
    user_code = user_info['usercode'] if user_info else None
    order_by = request.query_params.get("order_by", "latest")
    subscribed_videos = request.query_params.get("subscribed", "false") == "true"

    query = db.query(
        Fairytale.ft_code.label("id"),
        Fairytale.ft_name.label("url"),
        Fairytale.ft_title.label("title"),
        Fairytale.ft_like.label("ft_like"),
        User.user_name.label("name"),
        Profile.profile_name.label("img"),  # User.profile -> Profile.profile_name으로 수정
        User.user_code.label("author_id"),
        (db.query(Like).filter(Like.user_code == user_code, Like.ft_code == Fairytale.ft_code).exists()).label("liked")
    ).join(User, Fairytale.user_code == User.user_code).join(Profile, User.user_code == Profile.user_code)

    if subscribed_videos and user_code:
        subscriptions = db.query(Subscribe.user_code2).filter(Subscribe.user_code == user_code).subquery()
        query = query.filter(Fairytale.user_code.in_(subscriptions))
    
    if order_by == "popular":
        query = query.order_by(Fairytale.ft_like.desc())
    else:
        query = query.order_by(Fairytale.ft_code.desc())

    videos = query.limit(16).all()

    video_data = [
        {
            "id": video.id,
            "url": video.url if video.url else None,
            "name": video.name if video.name else "",
            "title": video.title if video.title else "",
            "ft_like": video.ft_like,
            "img": f"static/uploads/{video.img}" if video.img else "/static/uploads/basic.png",
            "liked": video.liked,
            "author_id": video.author_id
        }
        for video in videos
    ]


    profile_user_info, profile_image, follow_count, follower_count, total_likes = (None, "/static/uploads/basic.png", 0, 0, 0)
    if user_info:
        profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_code)

    return templates.TemplateResponse("main.html", {
        "request": request,
        "videos": video_data,
        "user_code": user_code,
        "profile_user_info": profile_user_info,
        "profile_image": profile_image,
        "follow_count": follow_count,
        "follower_count": follower_count,
        "total_likes": total_likes,
        "order_by": order_by,
        "subscribed_videos": subscribed_videos
    })


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
        Profile.profile_name.label("img"),
        User.user_code.label("author_id"),
        (db.query(Like).filter(Like.user_code == user_code, Like.ft_code == Fairytale.ft_code).exists()).label("liked")
    ).join(User, Fairytale.user_code == User.user_code).join(Profile, User.user_code == Profile.user_code).order_by(Fairytale.ft_code.desc()).limit(10).all()

    
    video_data = [
        {
            "id": video.id,
            "url": video.url if video.url else None,
            "name": video.name if video.name else "",
            "title": video.title if video.title else "",
            "ft_like": video.ft_like,
            "img": f"/static/uploads/{video.img}" if video.img else "/static/uploads/basic.png",
            "liked": video.liked,
            "author_id": video.author_id,
        }
        for video in videos
        
    ]

    profile_user_info, profile_image, follow_count, follower_count, total_likes = (None, "/static/uploads/basic.png", 0, 0, 0)
    if user_info:
        user_code = user_info['usercode']
        profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_code)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "videos": video_data,
        "user_code": user_code,
        "profile_user_info": profile_user_info,
        "profile_image": profile_image,
        "follow_count": follow_count,
        "follower_count": follower_count,
        "total_likes": total_likes
    })


async def get_profile_data(db: Session, user_code: int):
    profile_user = db.query(User).filter(User.user_code == user_code).first()
    profile_image = "/static/uploads/basic.png"
    follow_count = 0
    follower_count = 0
    total_likes = 0

    if profile_user:
        profile_user_info = {
            "user_code": profile_user.user_code,
            "user_name": profile_user.user_name,
            "email": profile_user.email,
            "profile": profile_user.profile,
        }
        profile = db.query(Profile).filter(Profile.user_code == user_code).first()
        profile_image = f"/static/uploads/{profile.profile_name}" if profile else profile_image
        follow_count = db.query(Subscribe).filter(Subscribe.user_code == user_code).count()
        follower_count = db.query(Subscribe).filter(Subscribe.user_code2 == user_code).count()
        user_fairytales = db.query(Fairytale).filter(Fairytale.user_code == user_code).all()
        total_likes = sum(fairy.ft_like for fairy in user_fairytales)
    else:
        profile_user_info = None

    return profile_user_info, profile_image, follow_count, follower_count, total_likes


# @app.get("/w_header", response_class=HTMLResponse)
# async def w_header(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
#     author_id = user_info['usercode']
#     return await get_profile(request, db, author_id, user_info)


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
    is_own_profile = (current_user_info['usercode'] == profile_user_info.user_code) if current_user_info else False
    is_following = db.query(Subscribe).filter_by(user_code=current_user_info['usercode'], user_code2=author_id).first() is not None

    profile_user_info_dict = {
        "user_code": profile_user_info.user_code,
        "user_name": profile_user_info.user_name,
        "email": profile_user_info.email,
        "profile": profile_user_info.profile,
    }

    # 팔로우 및 팔로워 수를 쿼리합니다.
    follow_count = db.query(Subscribe).filter(Subscribe.user_code == author_id).count()
    follower_count = db.query(Subscribe).filter(Subscribe.user_code2 == author_id).count()

    is_own_profile = current_user_info['usercode'] == author_id if current_user_info else False
    is_following = db.query(Subscribe).filter(Subscribe.user_code == current_user_info['usercode'], Subscribe.user_code2 == author_id).first() if current_user_info else False

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
        "is_own_profile": is_own_profile,
        "is_following": is_following,
        "follow_count": follow_count,
        "follower_count": follower_count

    })

@app.get("/upload", response_class=HTMLResponse)
async def display_form(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        user_code = user_info['usercode']
        
        # 기존 코드
        user_voices = db.query(Voice).filter(Voice.user_code == user_code).all()

        # 프로필 데이터 추가
        profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_code)

        return templates.TemplateResponse("upload.html", {
            "request": request,
            "user_voices": user_voices,
            "profile_user_info": profile_user_info,
            "profile_image": profile_image,
            "follow_count": follow_count,
            "follower_count": follower_count,
            "total_likes": total_likes
        })
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise
# 비디오 업로드 엔드포인트
@app.post("/upload_video")
async def upload_video(request: Request, db: Session = Depends(get_db)):
    return await upload.upload_video(request, db)


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
            story_title = story_content.split('\n')[0].replace("제목: ", "").replace("Title: ", "").replace('"', '').strip()
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
            audio_file_path, audio_name = predict.predict(selected_voice, story_content, language, speed)
            rmpath = f"static/audio/{audio_name}"
            if os.path.exists(rmpath) and os.path.isdir(rmpath):
                shutil.rmtree(rmpath)
                print(f"폴더 {rmpath}이(가) 삭제되었습니다.")
            else:
                print(f"폴더 {rmpath}이(가) 존재하지 않거나 디렉토리가 아닙니다.")

        paragraphs = story_content.split('\n\n')
        prompt_paragraphs = [
            f"{changeImg} {paragraph}" for paragraph in paragraphs
        ]
        print("prompt_paragraphs : ", prompt_paragraphs)

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

        # video_url.json 파일 업데이트
        video_data = {
            "video_url": final_output_file,
            "story_title": story_title
        }
        with open('video_url.json', 'w') as f:
            json.dump(video_data, f)

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

        print(f"Final video file 뜨냐: {final_output_file}") 
        return final_output_file
    except Exception as e:
        print(f"비디오 생성 중 오류가 발생하였습니다: {e}")
        raise

@app.get("/story_view", response_class=HTMLResponse)
async def story_view(request: Request, video_url: str, story_title: str, story_content: str, db: Session = Depends(get_db)):
    user_info = request.session.get('user')
    user_code = user_info['usercode'] if user_info else None

    profile_user_info, profile_image, follow_count, follower_count, total_likes = (None, "/static/uploads/basic.png", 0, 0, 0)
    if user_info:
        profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_code)

    return templates.TemplateResponse("story.html", {
        "request": request,
        "video_url": video_url,
        "story_title": story_title,
        "story_content": story_content,
        "profile_user_info": profile_user_info,
        "profile_image": profile_image,
        "follow_count": follow_count,
        "follower_count": follower_count,
        "total_likes": total_likes
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
    video_path = fairytale.ft_name

    # 절대 경로로 변환
    abs_video_path = os.path.abspath(video_path)

    # 디버깅 정보 출력
    print(f"video_path: {video_path}")
    print(f"abs_video_path: {abs_video_path}")
    print(f"Current working directory: {os.getcwd()}")

    # 데이터베이스에서 비디오 레코드 삭제
    db.delete(fairytale)
    db.commit()

    # 파일 시스템에서 비디오 파일 삭제
    if os.path.exists(abs_video_path):
        print(f"File exists: {abs_video_path}, attempting to delete.")
        try:
            os.remove(abs_video_path)
            print("File successfully deleted.")
            return {"message": "비디오 및 관련 좋아요가 삭제되었습니다!"}
        except Exception as e:
            print(f"Error deleting video file: {e}")
            raise HTTPException(status_code=500, detail=f"비디오 파일 삭제 중 오류가 발생했습니다: {str(e)}")
    else:
        print(f"File does not exist: {abs_video_path}")
        return {"message": "비디오 및 관련 좋아요가 삭제되었습니다! (파일이 이미 존재하지 않음)"}

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


@app.post("/follow/{user_code2}")
async def follow_user(user_code2: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    
    existing_subscription = db.query(Subscribe).filter_by(user_code=user_code, user_code2=user_code2).first()
    if existing_subscription:
        raise HTTPException(status_code=400, detail="Already following this user")

    new_subscription = Subscribe(user_code=user_code, user_code2=user_code2)
    db.add(new_subscription)
    db.commit()
    return {"message": "팔로우 되었습니다"}

@app.post("/unfollow/{user_code2}")
async def unfollow_user(user_code2: int, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    
    existing_subscription = db.query(Subscribe).filter_by(user_code=user_code, user_code2=user_code2).first()
    if not existing_subscription:
        raise HTTPException(status_code=400, detail="Not following this user")

    db.delete(existing_subscription)
    db.commit()
    return {"message": "팔로우 취소 되었습니다"}


@app.get("/gudog", response_class=HTMLResponse)
async def show_following_users(request: Request, db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = request.query_params.get("user_code")
    if not user_code:
        user_code = user_info['usercode']
    
    print("Requested user_code:", user_code)
    
    following = db.query(Subscribe).filter(Subscribe.user_code == user_code).all()
    following_users = []
    for follow in following:
        user = db.query(User).filter(User.user_code == follow.user_code2).first()
        profile = db.query(Profile).filter(Profile.user_code == user.user_code).first()
        profile_image = f"/static/uploads/{profile.profile_name}" if profile else "/static/uploads/basic.png"
        following_users.append({
            "user_code": user.user_code,
            "user_name": user.user_name,
            "profile_image": profile_image
        })
    
    print("Following users:", following_users)

    profile_user_info, profile_image, follow_count, follower_count, total_likes = await get_profile_data(db, user_info['usercode'])

    return templates.TemplateResponse("gudog.html", {
        "request": request,
        "following_users": following_users,
        "profile_user_info": profile_user_info,
        "profile_image": profile_image,
        "follow_count": follow_count,
        "follower_count": follower_count,
        "total_likes": total_likes
    })

@app.delete("/delete_existing_voice")
async def delete_existing_voice(db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    user_code = user_info['usercode']
    voice = db.query(Voice).filter(Voice.user_code == user_code).first()

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

    return {"message": "기존 목소리가 삭제되었습니다."}

# if __name__ == "__main__":
#     # 클라이언트에서 인터넷으로 다이렉트 요청할 때
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000
#     )
    # nginx가 앞단에 있으면
    import uvicorn
    uvicorn.run(app, port=8000
    )