import os
import shutil
import torch
import tempfile
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import requests

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
    target_dir = "static/audio"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    target_se, audio_name = se_extractor.get_se(
        audio,
        tone_color_converter,
        target_dir=target_dir,
        vad=False,
    )

    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id

    src_path = os.path.join(target_dir, "tmp.wav")
    out_path = os.path.join(tempfile.gettempdir(), "out.wav")

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

    # 변환된 파일을 "static/audio/m1.mp3"로 저장
    final_path = os.path.join(target_dir, "m1.mp3")
    shutil.copyfile(out_path, final_path)
    return final_path

# 인터페이스 코드
# iface = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Audio(source="upload", type="filepath", label="Upload Audio"),
#         gr.Textbox(lines=2, placeholder="Enter Text Here...", label="Input Text"),
#         gr.Dropdown(choices=["EN_NEWEST", "EN", "ES", "FR", "ZH", "JP", "KR"], value="KR", label="Select Language"),
#         gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed"),
#     ],
#     outputs=gr.Audio(label="Synthesised Audio", autoplay=True),
#     title="OpenVoice Text-to-Speech",
#     description="Convert your text to speech using OpenVoice enhanced by specific audio tones."
# )

# if __name__ == "__main__":
#     iface.launch()
