import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import torch
from collections import deque

from ultralytics import YOLO
from model.RepVGGLSTM import RepVGGFeatureExtractor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class_names = ['Rappel 60 & H',
              'Feu vert & A',
              'Avertissement & A',
              'Feu jaune clignotant & C',
              'Feu jaune clignotant & A',
              'Feu blancs &  ID3',
              'Avertissement & rappel 60 & H',
              'Feu rouge clignotant & F']

box_model = YOLO(ROOT / 'yolo-weights/yolov8n-single-cls/weights/best.pt')
class_model = RepVGGFeatureExtractor(hidden_size=256, num_classes=8)

class_model_wt = ROOT/'checkpoints/exp10/best.pt'
class_model.load_state_dict(torch.load(class_model_wt)['model_state_dict'])
class_model.eval()

# Define path to video file
source = str(ROOT / 'video_stream/clip5/video/clip5_fps20.mp4')

cap = cv2.VideoCapture(source)
id_to_buffer = {}
id_to_class_label = {}
timesteps = 16
previous_boxes = None
previous_ids = None


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = box_model.track(frame, imgsz = (2048, 1536) ,persist=True, agnostic_nms = True)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        if results[0].boxes and results[0].boxes.xyxy.cpu() is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        else:
            boxes = None

        if results[0].boxes and results[0].boxes.id.cpu() is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            ids = None
        # buffers = [[] for _ in range(len(boxes))]
        # print(len(buffers))
        if ids is not None:
            for id in ids:
                if id not in id_to_buffer:
                    id_to_buffer[id] = deque(maxlen=timesteps)
        if boxes is None or ids is None:
            print("None detected! Using previous information.")
            boxes = previous_boxes
            ids = previous_ids
        else:
            previous_boxes = boxes
            previous_ids = ids

        for i, (box, id) in enumerate(zip(boxes, ids)):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            crop_img = frame[box[1]:box[3], box[0]:box[2]]

            crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img_resized = cv2.resize(crop_img_rgb, (224, 224))  # Adjust the size as required by the model
            crop_img_normalized = crop_img_resized / 255.0  # Normalize pixel values to [0, 1]
            crop_img_tensor = torch.tensor(crop_img_normalized).unsqueeze(0).permute(0, 3, 1, 2).float()

            id_to_buffer[id].append(crop_img_tensor.squeeze(0))
            # if id not in id_to_buffer:
            #     id_to_buffer[id] = []
            # id_to_buffer[id].append(crop_img_tensor.squeeze(0))

            # Perform classification using class_model
            if len(id_to_buffer[id]) == timesteps:
                buffer_tensor = torch.stack(list(id_to_buffer[id])).unsqueeze(0)
                with torch.no_grad():
                    class_model_output = class_model(buffer_tensor)
                    probabilities = torch.softmax(class_model_output, dim=1)
                    _, predicted_class = torch.max(probabilities, 1)
                    class_label_index = int(predicted_class.item())
                    class_label_name = class_names[class_label_index]
                    id_to_class_label[id] = class_label_name

            class_label = id_to_class_label.get(id, None)
            if class_label is not None:
                cv2.putText(
                    frame,
                    f"Id{id}_Class{class_label}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        # Display the annotated frame
        cv2.imshow("Signal Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()