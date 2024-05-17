from fastapi import FastAPI, Form, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
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
from sqlalchemy.orm import Session
from models import User, Fairytale
from dependencies import get_db
from starlette.status import HTTP_303_SEE_OTHER
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import upload

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SECRET_KEY'))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def get_current_user(request: Request):
    user_info = request.session.get('user')
    if not user_info:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_info



@app.get("/", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
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
async def display_form(request: Request, user_info: dict = Depends(get_current_user)):
    print("사용자 정보 나오는지 확인", user_info)
    try:
        return templates.TemplateResponse("profile.html", {"request": request, "user_info": user_info})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params={'access_type': 'offline', 'prompt': 'consent'},
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri=os.getenv('GOOGLE_REDIRECT_URI'),
    client_kwargs={'scope': 'openid email profile https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/youtube.readonly'},
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs'
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

    existing_user = db.query(User).filter(User.email == email).first()

    if existing_user:
        request.session['user'] = {
            "usercode": existing_user.user_code,
            "name": existing_user.user_name,
            "email": existing_user.email,
            "picture": existing_user.profile
        }
        return RedirectResponse(url='/', status_code=HTTP_303_SEE_OTHER)

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

    request.session['user'] = {
        "usercode": new_user.user_code,
        "name": new_user.user_name,
        "email": new_user.email,
        "picture": new_user.profile
    }

    return RedirectResponse(url='/', status_code=HTTP_303_SEE_OTHER)

@app.get("/friends", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("friends.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.post("/story", response_class=HTMLResponse)
async def create_story(request: Request, keywords: str = Form(...), selected_voice: str = Form(...), db: Session = Depends(get_db), user_info: dict = Depends(get_current_user)):
    try:
        img_dir = "static/img"
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 어린이 동화를 만드는 AI야."},
                {"role": "user", "content": f"{keywords} 이 문자을 사용해서 동화 제목을 제목: 이렇게 지어주고, 동화 이야기를 공백포함 300자로 작성해주고, 4단락으로 나눠줘"}
            ]
        )

        if completion.choices:
            story_content = completion.choices[0].message.content
            story_title = story_content.split('\n')[0].replace("제목: ", "")
        else:
            story_content = "텍스트를 다시 입력해주세요!"

        new_story = Fairytale(
            user_code=user_info['usercode'],
            ft_title=story_title,
            ft_date=datetime.datetime.utcnow(),
            ft_name=story_content,
            ft_like=0
        )
        db.add(new_story)
        db.commit()

        language = "KR"
        speed = 1.0

        if selected_voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
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

        image_paths = []
        paragraphs = story_content.split('\n\n')
        delay_seconds = 15

        response = client.images.generate(
            model="dall-e-3",
            prompt=f"""
            "Create a four-panel fairytale image in a square digital art style. The layout is as follows: the top left corner captures the first part, the top right corner captures the second part, and the bottom left corner captures the third part. , the lower right corner shows the fourth part. The style should be vibrant and attractive, with no spaces between cuts to create a seamless visual narrative.”
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

            img = Image.open(img_dest)
            crop_sizes = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]

            for idx, (left, upper, right, lower) in enumerate(crop_sizes):
                cropped_image = img.crop((left, upper, right, lower))
                panel_filename = f"a{idx + 1}.jpg"
                panel_dest = os.path.join("static", "img", panel_filename)
                if os.path.exists(panel_dest):
                    os.remove(panel_dest)
                cropped_image.save(panel_dest)

        await create_video()

        return templates.TemplateResponse("story.html", {
            "request": request,
            "story_content": story_content,
            "story_title": story_title,
            "audio_file_path": audio_file_path,
            "image_paths": image_paths
        })
    except Exception as e:
        return f"스토리 생성 및 비디오 생성 중 오류가 발생하였습니다: {e}"

async def create_video():
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

        def image_to_video(images, output_file='output.mp4', fps=24):
            clip = ImageSequenceClip(images, fps=fps)
            clip.write_videofile(output_file, codec='libx264')

        def overlay_image_and_audio_on_video(video_file, audio_file, output_file='final_output.mp4'):
            video_clip = VideoFileClip(video_file)
            audio_clip = AudioFileClip(audio_file)
            final_clip = CompositeVideoClip([video_clip.set_audio(audio_clip)])
            final_clip.write_videofile(output_file, codec='libx264')

        def main():
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

            image_to_video(all_zoomed_images, 'static/output.mp4', fps=24)
            overlay_image_and_audio_on_video('static/output.mp4', 'static/audio/m1.mp3', 'static/final_output.mp4')

        print('sssss')
        main()
        return "비디오 생성이 완료되었습니다."
    except Exception as e:
        return f"비디오 생성 중 오류가 발생하였습니다: {e}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
