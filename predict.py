import os
import shutil
import torch
import tempfile
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import requests
import uuid

# 모델 설정 및 다운로드
MODEL_URL = "https://weights.replicate.delivery/default/myshell-ai/OpenVoice-v2.tar"
MODEL_CACHE = "model_cache"

def setup_model():
    if not os.path.exists(MODEL_CACHE):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_CACHE, 'wb') as f:
            f.write(response.content)
        print("Model downloaded and set up.")

setup_model()

ckpt_converter = f"{MODEL_CACHE}/checkpoints_v2/converter"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

def predict(audio, text, language="KR", speed=1.2):
    print(language, "languagelanguagelanguagelanguage")
    print(audio, "audioaudioaudioaudioaudioaudioaudioaudioaudioaudio")
    print("11111111111111111111111")
    
    target_dir = "static/audio"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    print(tone_color_converter, "tone_color_converter")
    print(target_dir,"target_dir")
    target_se, audio_name = se_extractor.get_se(
        audio,
        tone_color_converter,
        target_dir=target_dir,
        vad=False,
    )
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    print("222222222222222222222")

    # 고유한 파일 이름 생성
    unique_id = str(uuid.uuid4())
    src_path = os.path.join(target_dir, f"tmp_{unique_id}.wav")
    out_path = os.path.join(tempfile.gettempdir(), f"out_{unique_id}.wav")
    print("333333333333333333333333333333333")
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace("_", "-")

        print(speaker_key, "speaker_keyspeaker_keyspeaker_keyspeaker_key")

        source_se = torch.load(
            f"{MODEL_CACHE}/checkpoints_v2/base_speakers/ses/{speaker_key}.pth",
            map_location=device,
        )
        model.tts_to_file(text, speaker_id, src_path, speed=speed)

        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message=encode_message,
        )
    # 변환된 파일을 고유한 이름으로 저장
    final_path = os.path.join(target_dir, f"m1_{unique_id}.mp3")
    shutil.copyfile(out_path, final_path)
    print("44444444444444444444444444444444444")
    
    print(audio_name, "audio_name")
    print(src_path, "src_path")
    
    if os.path.exists(src_path):
        os.remove(src_path)
    return final_path, audio_name
