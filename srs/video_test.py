import cv2
import os
from segmentation_models_pytorch import Unet
import torch
import albumentations as A
from PIL import Image
import numpy as np

# Шаг 1: Извлечение кадров
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"Извлечено {frame_count} кадров.")

# Шаг 2: Сегментация кадров
def segment_frame(model, frame, transform):
    transformed = transform(image=frame)
    image = transformed['image']
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    return mask

def process_frames(input_dir, output_dir, model, transform):
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_file in os.listdir(input_dir):
        frame_path = os.path.join(input_dir, frame_file)
        frame = np.array(Image.open(frame_path))
        
        mask = segment_frame(model, frame, transform)
        mask_path = os.path.join(output_dir, frame_file)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

# Шаг 3: Создание видео
def create_video_from_frames(frame_dir, output_video_path, fps=30):
    frame_files = sorted(os.listdir(frame_dir))
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Видео сохранено: {output_video_path}")

# Основной скрипт
if __name__ == "__main__":
    # Настройки
    input_video = os.path.join("diploma", "dataset", "video", "video2.mp4")
    frames_dir = os.path.join("diploma", "dataset", "framed_video")
    segmented_frames_dir = os.path.join("diploma", "dataset", "segm_frsmed_video")
    output_video =os.path.join("diploma", "output_video.mp4")
    
    # Шаг 1: Извлечение кадров
    extract_frames(input_video, frames_dir)
    
    # Шаг 2: Сегментация
    model = Unet(encoder_name="resnet34", classes=2, activation=None)
    model.load_state_dict(torch.load(os.path.join("diploma", "trained_model", "model_epoch_10.pth")))
    model.eval()
    
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    process_frames(frames_dir, segmented_frames_dir, model, transform)
    
    # Шаг 3: Создание видео
    create_video_from_frames(segmented_frames_dir, output_video)