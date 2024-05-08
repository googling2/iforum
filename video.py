from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip, CompositeVideoClip, CompositeAudioClip
from moviepy.editor import AudioFileClip
import os

def create_image(image_file, output_size=(512, 512)):
    """지정된 크기로 이미지 파일을 열고 크기를 조정합니다."""
    image = Image.open(image_file)
    image = image.resize(output_size, Image.LANCZOS)
    return image

def create_zoom_frames(image, duration=6, fps=24, final_scale=1.3):
    """주어진 이미지에 대해 지정된 기간과 fps로 줌 효과의 프레임을 생성합니다."""
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
    """지정된 프레임 속도로 이미지 배열 목록에서 비디오를 만듭니다."""
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_file, codec='libx264')

def overlay_image_and_audio_on_video(video_file, audio_file, output_file='final_output.mp4'):
    """비디오에 오디오 트랙을 오버레이하고 지정된 출력 파일로 내보냅니다."""
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    final_clip = CompositeVideoClip([video_clip.set_audio(audio_clip)])
    final_clip.write_videofile(output_file, codec='libx264')

def main():
    """이미지를 확대하는 비디오로 처리하고 오디오를 오버레이하는 메인 함수입니다."""
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

if __name__ == "__main__":
    main()