# i-forum 프로젝트

## 기간
- 2024.04.23 ~ 2024.06.04

## 구성
- 4명

## 소개
- i-forum은 AI를 활용한 동화 제작 플랫폼으로, 사용자들이 AI 모델을 통해 창의적이고 재미있는 동화를 쉽게 생성할 수 있도록 돕습니다. 이 프로젝트는 부모님의 목소리로 동화를 읽어주는 기능과 유튜브 자동 업로드 기능을 포함하여, 아이들이 쉽게 조작할 수 있는 UI를 제공합니다.

## 목적
1. AI 모델을 활용하여 창의적이고 재미있는 동화를 쉽게 생성
2. 부모님의 목소리로 동화를 읽어주는 기능
3. 유튜브 자동 업로드 버튼 구현을 통한 접근성 증가
4. 아이들도 쉽게 조작할 수 있도록 UI 제공

## 주요 기능
1. **AI 모델을 활용한 동화, 이미지, 음성 생성**
   - GPT 3.5 API를 사용하여 동화를 생성하고, DALL-e-3 API를 통해 동화에 맞는 이미지를 생성합니다.
   - Open Voice V2를 통해 동화의 음성을 부모님의 목소리로 생성합니다.

2. **유튜브 자동 업로드**
   - 생성된 동화를 영상으로 만든 후, 유튜브 API를 통해 자동으로 업로드하는 기능을 제공합니다.

3. **사용자 친화적인 UI**
   - 아이들이 쉽게 조작할 수 있도록, 보기 쉽고 직관적인 UI를 제공합니다.

## 맡은 역할
- **GPT 3.5 API**를 사용하여 동화 생성 및 프롬프트 제작
- **DALL-e-3 API**를 사용하여 동화 사진 생성
- **Fast API**를 활용하여 좋아요, 팔로우 기능, 메인페이지 및 마이페이지 구현

## 사용 언어 및 개발 환경
- **Language**: Python, JavaScript
- **DB**: MySQL, SQLite
- **Front end**: HTML, CSS
- **Back end**: Fast API, OpenAI API, Open Voice V2
- **Tool**: VS Code, GitHub, Git, Slack, Notion
- **Cloud & DevOps**: Amazon S3

## 추가 기능 설명
1. **목소리, 그림체 스타일, 배경음악 선택** 후 동화 제목을 입력하면, AI가 동화를 생성합니다.
2. **영상 생성 후 유튜브 업로드 버튼**을 통해 동화를 유튜브에 자동으로 업로드할 수 있습니다.

## 이미지
![image](https://github.com/user-attachments/assets/8c8c0d19-b7a7-4e29-9b6d-d0384f78bb2d)
1. 목소리, 그림체 스타일, 
배경음악을 선택하고
동화 제목을 쓰면 동화를 생성합니다
2. 영상이 만들어진 후 유튜브 업로드 버튼으로 자동 업로드 가능
