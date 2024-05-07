from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import openai
from dotenv import load_dotenv
import os
import urllib.request
import time
import asyncio
import shutil  # 디렉토리 삭제에 사용

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)



@app.get("/home", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/upload", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("upload.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/profile", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("profile.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/login", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("login.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise

@app.get("/friends", response_class=HTMLResponse)
async def display_form(request: Request):
    try:
        return templates.TemplateResponse("friends.html", {"request": request})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise




@app.post("/story", response_class=HTMLResponse)
async def create_story(request: Request, keywords: str = Form(...), selected_voice: str = Form(...)):
    # 이미지 디렉토리 초기화
    img_dir = "static/img"
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)  # 디렉토리 삭제
    os.makedirs(img_dir, exist_ok=True)  # 새 디렉토리 생성

    # GPT-3.5로 스토리 생성
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 어린이 동화를 만드는 AI야."},
            {"role": "user", "content": f"{keywords} 이 단어들을 사용해서 동화 이야기를 공백포함 300자로 작성해주고, 2단락으로 나눠줘"}
        ]
    )


    # 스토리 콘텐츠 확인
    if completion.choices:
        story_content = completion.choices[0].message.content
    else:
        story_content = "텍스트를 다시 입력해주세요!"


    print("동화내용 나오는지 확인 : ",story_content)


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

    # 이미지 생성 및 저장
    image_paths = []
    paragraphs = story_content.split('\n\n')
    rate_limit_per_minute = 5  # OpenAI 이미지 API 제한
    delay_seconds = 15
    # 60 / rate_limit_per_minute  # 15초

    for idx, paragraph in enumerate(paragraphs):
        if idx > 0:
            await asyncio.sleep(delay_seconds)  # 요청 사이 지연

        response = client.images.generate(
            model="dall-e-3",
            prompt=f"please draw  digital art style, {paragraph}",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        if response.data:
            image_url = response.data[0].url
            img_filename = f"a{idx + 1}.jpg"
            img_dest = os.path.join("static", "img", img_filename)
            if os.path.exists(img_dest):
                os.remove(img_dest)
            urllib.request.urlretrieve(image_url, img_dest)
            image_paths.append(img_dest)

    # 결과 템플릿 렌더링
    return templates.TemplateResponse("story.html", {
        "request": request,
        "story_content": story_content,
        "audio_file_path": audio_file_path,
        "image_paths": image_paths
    })


@app.post("/create_video", response_class=HTMLResponse)
async def create_video(request: Request):
    # 비동기적으로 video.py 실행
    process = await asyncio.create_subprocess_exec(
        "python", "video.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # 프로세스가 완료될 때까지 대기
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        final_output = "static\\final_output.mp4"
        return templates.TemplateResponse("video_created.html", {"request": request, "video_url": final_output})
    else:
        # 프로세스 실행 중 오류가 발생한 경우 처리
        error_message = f"{stderr.decode()}, 무슨 에러러러러????"
        # return templates.TemplateResponse("video_error.html", {"request": request, "error_message": error_message})
        return "에러입니다."
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
