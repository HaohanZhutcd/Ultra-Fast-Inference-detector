import os


def rename_files(src_dir):
    images_folder = os.path.join(src_dir, "images")
    labels_folder = os.path.join(src_dir, "labels")

    images_files = os.listdir(images_folder)
    labels_files = os.listdir(labels_folder)

    def modify_filename(files, folder_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1]
            if file_ext == ".jpg" or file_ext == ".png" or file_ext == ".txt":
                if "-" in filename:
                    new_filename = filename.split("-")[-1]
                    new_filename = os.path.splitext(new_filename)[0] + file_ext
                    old_path = os.path.join(folder_path, filename)
                    new_path = os.path.join(folder_path, new_filename)
                    os.rename(old_path, new_path)
                    # if os.path.exists(new_path) and new_path != old_path:
                    #     os.remove(old_path)

    modify_filename(images_files, images_folder)
    modify_filename(labels_files, labels_folder)


if __name__ == '__main__':
    source_dir = f'D:\Download\project-11-at-2023-08-05-14-55-2b5e9c1c'
    rename_files(source_dir)