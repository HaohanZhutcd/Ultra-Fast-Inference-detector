import os
import shutil


def create_labels_folder_structure(images_folder, labels_folder):
    image_subfolders = [subfolder for subfolder in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, subfolder))] # ['detect', 'test', 'train']

    for subfolder in image_subfolders:
        image_subfolder_path = os.path.join(images_folder, subfolder)
        labels_subfolder_path = os.path.join(labels_folder, subfolder)
        if not os.path.exists(labels_subfolder_path):
            os.makedirs(labels_subfolder_path)
            print("Create mapping folder in labels done")
        else:
            print("Mapping folder '{}' exist".format(subfolder))


def map_files_to_labels(src_dir):
    images_folder = os.path.join(src_dir, "images")
    labels_folder = os.path.join(src_dir, "labels")

    create_labels_folder_structure(images_folder, labels_folder)

    print(".txt files are being migrated")
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_file_path = os.path.join(root, file)
                image_relative_path = os.path.relpath(image_file_path, images_folder)

                labels_file_path = os.path.join(labels_folder, image_relative_path)
                labels_file_path = os.path.splitext(labels_file_path)[0] + ".txt"

                labels_folder_path = os.path.dirname(labels_file_path)
                os.makedirs(labels_folder_path, exist_ok=True)

                source_txt_path = os.path.join(labels_folder, os.path.splitext(file)[0] + ".txt")
                if os.path.exists(source_txt_path):
                    shutil.move(source_txt_path, labels_file_path)
    print("mapping done")


if __name__ == "__main__":
    src_dir = 'C:/Users/zhuha/Desktop/YOLO-FRSign/FRSignData'

    map_files_to_labels(src_dir)
