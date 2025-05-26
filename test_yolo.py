from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import torch


model_path = "/home/HiPose_Online_mahjong/ultralytics/runs/detect/mahjong/weights/best.pt"
model = YOLO(model=model_path)
model.eval()
model.to(device='cuda', dtype=torch.bfloat16)

rgb_folder = "/home/HiPose_Online_mahjong/data/rgb"
images = sorted(
    [
        os.path.join(rgb_folder, file)
        for file in os.listdir(rgb_folder)
        if file.endswith(('.jpg', '.png'))
    ]
)

# Define the output video path and settings
video_path = "/home/HiPose_Online_mahjong/output_video.avi"
frame_width, frame_height = cv2.imread(images[0]).shape[1], cv2.imread(images[0]).shape[0]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # Frames per second
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width * 2, frame_height))

for image_path in tqdm(images):
    image_array = cv2.imread(image_path)
    results = model.predict(image_path, conf=0.6)
    id2labels = results[0].names
    cls = results[0].boxes.cls
    conf = results[0].boxes.conf
    xyxy = results[0].boxes.xyxy
    # Create a copy of the original image to draw all boxes and labels
    annotated_image = image_array.copy()

    for i, box in enumerate(xyxy):
        label = id2labels[int(cls[i])]
        confidence = conf[i].item()
        x1, y1, x2, y2 = map(int, box)
        
        # Draw the box and text on the annotated image
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{label}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 - 5 if y1 - 5 > 0 else y1 + text_size[1] + 5
        # Map confidence to a color gradient from red (low confidence) to green (high confidence)
        confidence_color = (0, int(confidence * 255), int((1 - confidence) * 255))
        cv2.putText(annotated_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)

    # Concatenate the original image and the annotated image horizontally
    combined_frame = cv2.hconcat([image_array, annotated_image])

    # Write the frame to the video
    video_writer.write(combined_frame)

# Release the video writer
video_writer.release()
