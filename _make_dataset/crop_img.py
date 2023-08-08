import os
import cv2
import shutil
import numpy as np
import yaml



def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_frame_sequence(split_folder, clip_folder):
    frames = []
    clip_folder_path = os.path.join(root_dir, "images", split_folder, clip_folder)
    frame_files = os.listdir(clip_folder_path)
    frame_files.sort()
    for frame_file in frame_files:
        frame_path = os.path.join(clip_folder_path, frame_file)
        frames.append(frame_path)
    return frames


def compute_sift_features(frame_path):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_sift_features(descriptors1, descriptors2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


def compute_distance(bbox1, bbox2):
    center_x1 = (bbox1[0] + bbox1[2]) / 2
    center_y1 = (bbox1[1] + bbox1[3]) / 2
    center_x2 = (bbox2[0] + bbox2[2]) / 2
    center_y2 = (bbox2[1] + bbox2[3]) / 2

    distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance


def crop_and_save_image(frame_path, bbox, class_name, object_id, clip_folder, frame_index, target_folder):
    img = cv2.imread(frame_path)
    cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    save_path = os.path.join(target_folder, f"{clip_folder}_{object_id:02d}_{frame_index:03d}.png")
    cv2.imwrite(save_path, cropped_img)


def compute_bbox_center(bbox):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y


def compute_bbox_distance(bbox1, bbox2):
    center_x1, center_y1 = compute_bbox_center(bbox1)
    center_x2, center_y2 = compute_bbox_center(bbox2)
    distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance


max_distance = 15  # Set the distance threshold for matching objects


# def process_clip(split_folder, clip_folder, class_names):
#     global object_counter  # Use the global object_counter variable
#
#     clip_images_path = os.path.join(root_dir, "images", split_folder, clip_folder)
#     clip_labels_path = os.path.join(root_dir, "labels", split_folder, clip_folder)
#
#     prev_objects = {}  # Dictionary to store previous frame's objects with their IDs
#     last_assigned_ids = {}  # Dictionary to store the last assigned object ID for each class
#     frame_index = 1
#     object_counter = 1  # Global variable to keep track of the object ID
#
#     for label_file in os.listdir(clip_labels_path):
#         label_path = os.path.join(clip_labels_path, label_file)
#         image_path = os.path.join(clip_images_path, label_file.replace(".txt", ".png"))
#
#         # Read class and bbox information from label file
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
#
#         objects_in_frame = {}  # Dictionary to store objects in the current frame with their IDs
#
#         for line in lines:
#             class_id, center_x, center_y, width, height = map(float, line.strip().split())
#             class_name = class_names[int(class_id)]
#
#             # Compute bounding box coordinates
#             left = int((center_x - width / 2) * image_width)
#             top = int((center_y - height / 2) * image_height)
#             right = int((center_x + width / 2) * image_width)
#             bottom = int((center_y + height / 2) * image_height)
#             bbox = (left, top, right, bottom)
#
#             # Check if the current object matches any of the previous objects based on bbox distance
#             matched_object_id = None
#             if prev_objects and class_name in last_assigned_ids:
#                 for prev_object_id, (prev_class_name, prev_bbox) in prev_objects.items():
#                     if class_name == prev_class_name:
#                         distance = compute_bbox_distance(bbox, prev_bbox)
#                         if distance < max_distance:
#                             matched_object_id = prev_object_id
#                             break
#
#             if matched_object_id is not None:
#                 # Assign the same ID to the current object as the matched previous object
#                 object_id = matched_object_id
#             else:
#                 # If no matching object found or first appearance, assign a new ID to the current object
#                 object_id = last_assigned_ids.get(class_name, object_counter)
#                 last_assigned_ids[class_name] = object_id
#                 object_counter += 1  # Increment the global object_counter
#
#             # Store the current object data in the dictionary for future comparison
#             objects_in_frame[object_id] = (class_name, bbox)
#
#         # Crop and save the images for all objects in the current frame
#         for object_id, obj_data in objects_in_frame.items():
#             class_name, bbox = obj_data
#             target_folder = os.path.join(crop_imgs_dir, class_name)
#             create_folder(target_folder)
#             crop_and_save_image(image_path, bbox, class_name, object_id, clip_folder, frame_index, target_folder)
#
#         # Update previous objects with the current frame's objects data
#         prev_objects = objects_in_frame
#
#         frame_index += 1


def process_clip(split_folder, clip_folder, class_names):
    global object_counter  # Use the global object_counter variable

    clip_images_path = os.path.join(root_dir, "images", split_folder, clip_folder)
    clip_labels_path = os.path.join(root_dir, "labels", split_folder, clip_folder)

    prev_objects = {}  # Dictionary to store previous frame's objects with their IDs
    last_assigned_ids = {}  # Dictionary to store the last assigned object ID for each class
    frame_index = 1
    object_counter = 1  # Global variable to keep track of the object ID

    for label_file in os.listdir(clip_labels_path):
        label_path = os.path.join(clip_labels_path, label_file)
        image_path = os.path.join(clip_images_path, label_file.replace(".txt", ".png"))

        # Read class and bbox information from label file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        objects_in_frame = {}  # Dictionary to store objects in the current frame with their IDs

        for line in lines:
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
            class_name = class_names[int(class_id)]

            # Compute bounding box coordinates
            left = int((center_x - width / 2) * image_width)
            top = int((center_y - height / 2) * image_height)
            right = int((center_x + width / 2) * image_width)
            bottom = int((center_y + height / 2) * image_height)
            bbox = (left, top, right, bottom)

            # Check if the current object matches any of the previous objects based on bbox distance
            best_match = None
            min_distance = float('inf')
            if prev_objects and class_name in last_assigned_ids:
                for prev_object_id, (prev_class_name, prev_bbox) in prev_objects.items():
                    if class_name == prev_class_name:
                        distance = compute_bbox_distance(bbox, prev_bbox)
                        if distance < min_distance:
                            min_distance = distance
                            best_match = prev_object_id

            if best_match is not None and min_distance < max_distance:
                # Assign the same ID to the current object as the matched previous object
                object_id = best_match
            else:
                # If no matching object found or first appearance, assign a new ID to the current object
                object_id = last_assigned_ids.get(class_name, object_counter)
                last_assigned_ids[class_name] = object_id
                object_counter += 1  # Increment the global object_counter

            # Store the current object data in the dictionary for future comparison
            objects_in_frame[object_id] = (class_name, bbox)

        # Crop and save the images for all objects in the current frame
        for object_id, obj_data in objects_in_frame.items():
            class_name, bbox = obj_data
            target_folder = os.path.join(crop_imgs_dir, class_name)
            create_folder(target_folder)
            crop_and_save_image(image_path, bbox, class_name, object_id, clip_folder, frame_index, target_folder)

        # Update previous objects with the current frame's objects data
        prev_objects = objects_in_frame

        frame_index += 1


if __name__ == "__main__":
    root_dir = "../data"
    crop_imgs_dir = os.path.join(root_dir, "clip6")
    yaml_file_path = "../data/FRSignData.yaml"

    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        class_names = yaml_data["names"]

    image_width, image_height = 2048, 1536
    max_distance = 15

    create_folder(crop_imgs_dir)

    # for split_folder in ["train"]:
    #     split_folder_path = os.path.join(root_dir, "images", split_folder)
    #     for clip_folder in os.listdir(split_folder_path):
    #         frame_sequence = get_frame_sequence(split_folder, clip_folder)
    #         process_clip(split_folder, clip_folder, class_names)
    process_clip('detect', 'clip6', class_names)