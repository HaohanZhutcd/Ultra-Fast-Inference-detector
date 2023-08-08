import os
import cv2

def images_to_video(image_folder, video_name, fps=20):
    image_files = []
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()

    frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()

if __name__ == "__main__":
    fps = 20
    image_folder_path = "../video_stream/clip5/images"
    output_video_name = f"../video_stream/clip5/video/clip5_fps{fps}.mp4"
    images_to_video(image_folder_path, output_video_name, fps=fps)
