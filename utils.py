# utils.py

import streamlit as st
import http.client
import urllib.request
import urllib.parse
import urllib.error
import json
import os
import random
import shutil
import re
import streamlit as st
from ultralytics import YOLO
import shutil
from datetime import datetime
import yaml
import plotly.express as px
from unidecode import unidecode

training_in_progress = False
downloading = False

def disable():
    st.session_state.disabled = True
    print("Nút tải xuống bị vô hiệu hóa")

def enable():
    if "disabled" in st.session_state and st.session_state.disabled == True:
        st.session_state.disabled = False
        print("Nút tải xuống được kích hoạt")


##################################
# STEP 1
##################################

def remove_vietnamese_characters(text):
    text = unidecode(text)
    return text

def create_class_mapping(yaml_file_path, mapping_file_path):
    try:
        with open(yaml_file_path, 'r', encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file)
            names = data.get("names", [])

            with open(mapping_file_path, 'w', encoding="utf-8") as mapping_file:
                for index, class_name in enumerate(names):
                    class_name = remove_vietnamese_characters(class_name)
                    mapping_file.write(f"{index} {class_name}\n")

        st.success("Class mapping created successfully!")
    except Exception as e:
        st.error(f"Error occurred while creating class mapping: {str(e)}")

def download_images(project_id, training_key):
    global downloaded_images
    # Tạo thư mục data nếu chưa tồn tại
    data_dir = "data/train"
    os.makedirs(data_dir, exist_ok=True)

    # Đường dẫn tới thư mục images
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    headers = {
        # Request headers  
        'Training-key': training_key,
    }

    # Số lượng tối đa mà bạn muốn tải xuống trong mỗi lần gọi API
    batch_size = 256

    # Số lượng hình ảnh đã tải xuống
    downloaded_images = 0

    # Danh sách các file đã tải xuống
    downloaded_files = []

    # Danh sách tên các lớp
    class_names = set()

    # Biến để theo dõi số thứ tự của hình ảnh
    image_counter = 1

    # Biến để kiểm tra trạng thái tải xuống
    downloading = False

    # Tạo từ điển ánh xạ từ tên lớp sang số
    class_to_index = {}

    while True:
        # Sử dụng batch_size để xác định số lượng hình ảnh trong mỗi lần tải xuống
        params = urllib.parse.urlencode({
            # Request parameters
            'iterationId': '',
            'tagIds': '',
            'orderBy': 'newest',
            'take': str(batch_size),
            'skip': str(downloaded_images),
        })

        try:
            conn = http.client.HTTPSConnection('southeastasia.api.cognitive.microsoft.com')
            conn.request("GET", f"/customvision/v3.3/Training/projects/{project_id}/images/tagged?%s" % params, "{body}", headers)
            response = conn.getresponse()
            data = response.read()

            # Định dạng dữ liệu JSON
            parsed_data = json.loads(data.decode('utf-8'))

            # Kiểm tra nếu không còn hình ảnh nào để tải xuống
            if not parsed_data:
                break

            for image in parsed_data:
                id = image["id"]
                original_image_uri = image["originalImageUri"]

                # Tải hình ảnh
                image_data = urllib.request.urlopen(original_image_uri).read()

                # Định dạng tên tệp theo thứ tự 00001, 00002, 00003,...
                image_filename = f"{image_counter:05d}.jpg"

                # Lưu hình ảnh vào thư mục images
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)

                regions = image["regions"]

                # Trích xuất tên các lớp từ nhãn và tạo ánh xạ từ tên lớp sang số
                for region in regions:
                    tag_name = region["tagName"]
                    class_names.add(tag_name)
                    if tag_name not in class_to_index:
                        class_to_index[tag_name] = len(class_to_index)

                label_file_path = os.path.join(labels_dir, f"{image_counter:05d}.txt")
                image_counter += 1

                # Ghi thông tin label và bounding box vào tệp văn bản
                with open(label_file_path, 'w') as label_file:
                    for region in regions:
                        tag_name = region["tagName"]
                        class_index = class_to_index[tag_name]
                        center_x = region["left"] + region["width"] / 2
                        center_y = region["top"] + region["height"] / 2
                        width = region["width"]
                        height = region["height"]
                        label_file.write(f"{class_index} {center_x} {center_y} {width} {height}\n")

                # Thêm tên file vào danh sách đã tải xuống
                downloaded_files.append(image_filename)

                # In ra thông báo khi có file được tải xuống
                if not downloading:
                    st.write("Files are being downloaded. Please wait...")
                    downloading = True


            # Cộng số lượng hình ảnh đã tải xuống
            downloaded_images += len(parsed_data)

            conn.close()
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

    # Hiển thị thông báo khi tải xuống hoàn thành
    st.success(f"{downloaded_images} images downloaded successfully!")

    # Tạo file data.yaml
    data_yaml_path = os.path.join("data", "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as data_yaml_file:
        data_yaml_file.write("train: ../train/images\n")
        data_yaml_file.write("val: ../valid/images\n")
        data_yaml_file.write("\n")
        data_yaml_file.write(f"nc: {len(class_names)}\n")
        data_yaml_file.write("names: " + str(list(class_names)))
    create_class_mapping("data/data.yaml", "data/mapping.txt")

    return downloaded_files


def split_train_valid(image_folder, label_folder, valid_ratio):
    # Tạo thư mục đích nếu chưa tồn tại
    valid_folder = 'data/valid'
    os.makedirs(valid_folder, exist_ok=True)
    
    valid_image_folder = os.path.join(valid_folder, 'images')
    valid_label_folder = os.path.join(valid_folder, 'labels')

    os.makedirs(valid_image_folder, exist_ok=True)
    os.makedirs(valid_label_folder, exist_ok=True)

    # Lấy danh sách các tệp hình ảnh trong thư mục gốc
    image_files = os.listdir(image_folder)

    # Số lượng hình ảnh cần tách
    num_images_to_move = int(valid_ratio * len(image_files))

    # Tạo danh sách ngẫu nhiên các tệp cần di chuyển
    random.shuffle(image_files)
    files_to_move = image_files[:num_images_to_move]

    # Di chuyển các tệp hình ảnh và nhãn tương ứng
    for file in files_to_move:
        image_path = os.path.join(image_folder, file)
        label_file = file.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_file)

        # Kiểm tra xem tệp nhãn có tồn tại không trước khi di chuyển
        if os.path.exists(label_path):
            # Di chuyển tệp hình ảnh
            shutil.move(image_path, os.path.join(valid_image_folder, file))

            # Di chuyển tệp nhãn
            shutil.move(label_path, os.path.join(valid_label_folder, label_file))
        else:
            print(f"Label file '{label_path}' not found. Skipping move operation.")

    return num_images_to_move

def clear_data_folder():
    data_folder = "data"

    # Kiểm tra xem thư mục tồn tại không
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        # Lặp qua các tệp và thư mục trong thư mục data
        for root, dirs, files in os.walk(data_folder):
            # Xóa tất cả các tệp trong thư mục hiện tại
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    print("Data folder cleared successfully!")

##################################
# STEP 2
##################################

# Function to save the best.pt file with a unique name and delete the runs directory
def save_best_model():
    # global new_best_name
    # Find the latest train folder
    train_folders = [folder for folder in os.listdir('runs/detect/') if folder.startswith('train')]
    if not train_folders:
        st.warning("No training folder found.")
        return

    # Filter out train folders without numbers
    train_folders_number = [folder for folder in train_folders if any(char.isdigit() for char in folder)]

    if not train_folders_number:
        latest_train_folder = 'train'
    else:
        # Extract numeric part of folder name
        folder_numbers = [int(re.search(r'\d+', folder).group()) for folder in train_folders_number]

        latest_train_folder_number = max(folder_numbers)
        latest_train_folder = f"train{latest_train_folder_number}"

    # Find the latest best.pt file
    best_files = [file for file in os.listdir(f'runs/detect/{latest_train_folder}/weights/') if file.startswith('best')]
    if not best_files:
        st.warning("No best.pt file found.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_best_name = f"best_{timestamp}.pt"
    
    # Rename best.pt to the new name
    os.rename(f'runs/detect/{latest_train_folder}/weights/best.pt', f'runs/detect/{latest_train_folder}/weights/{new_best_name}')

    # Move best.pt to weights folder with unique name
    shutil.move(f"runs/detect/{latest_train_folder}/weights/{new_best_name}", f"weights/{new_best_name}")
    
    st.success(f"Saved {new_best_name} successfully.")
    # Delete all files and folders in runs directory
    shutil.rmtree('runs')





# Main function to run the YOLO training
def run_yolo_training(epochs, imgsz, optimizer, lr0, lrf, patience, batch, dropout):
    # global results
    # Folder paths
    yaml_location = 'data/data.yaml'

    # Training
    model = YOLO('yolov8m.yaml').load('yolov8m.pt')
    st.write("Starting training...")
    st.write("*Note: This process cannot be interrupted until it is complete.*")

    results = model.train(epochs=epochs, imgsz=imgsz, data=yaml_location, optimizer=optimizer, lr0=lr0, lrf=lrf, patience=patience, dropout=dropout)

    # Notification when training completes
    st.info("Training completed.")
    st.write('Precision(B):', results.results_dict['metrics/precision(B)'])
    st.write('Recall(B):', results.results_dict['metrics/recall(B)'])
    st.write('mAP50(B):', results.results_dict['metrics/mAP50(B)'])
    st.write('mAP50-95(B):', results.results_dict['metrics/mAP50-95(B)'])

##################################
# STEP 3
##################################

# Path to the file containing class information
class_info_path = 'data/mapping.txt'

# Check if the mapping file exists, if not, create an empty file
if not os.path.exists(class_info_path):
    with open(class_info_path, "w", encoding="utf-8"):
        pass  # Create an empty file

# Read the mapping file to map class ID to class name
class_mapping = {}
with open(class_info_path, "r", encoding="utf-8") as mapping_file:
    lines = mapping_file.readlines()
    for line in lines:
        class_id, class_name = line.strip().split(" ", 1)
        class_mapping[class_id] = class_name

# Get the Light24 colormap
annotation_colormap = px.colors.qualitative.Light24

# Function to list all .pt files in the weights directory
def list_pt_files():
    weights_dir = "weights"
    return [f for f in os.listdir(weights_dir) if f.endswith(".pt")]

def get_color(class_id):
    """
    Get a color for the bounding box based on the class_id.
    """
    # Get the color from the Light24 colormap based on class_id
    color_index = class_id % len(annotation_colormap)
    color = annotation_colormap[color_index]

    # Convert HEX color to RGB
    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

    return color_rgb

# Dictionary to store information about detected objects
prediction_dict = {}