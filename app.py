# app.py
# streamlit run app.py --server.maxUploadSize=1028
import streamlit as st
import os
import zipfile
import cv2
import numpy as np
from ultralytics import YOLO
from utils import *
import shutil

# Set page configuration
st.set_page_config(
    page_title="Training Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Step 1: Import Database
def import_database():
    st.title("Step 1: Import Database")
    st.write('Importing images and bounding boxes from Custom Vision or from a Zip file')

    st.sidebar.title("Select Import Method")
    conversion_method = st.sidebar.radio("Choose method:", ("Import from Custom Vision", "Import from a ZIP file"))

    if conversion_method == "Import from Custom Vision":
        # if "disabled" not in st.session_state:
        #     st.session_state.disabled = False
        project_id = st.text_input("Enter project ID:")
        training_key = st.text_input("Enter training key:")
        valid_ratio = st.text_input("Enter validation split ratio (e.g., 0.2):")

        if not project_id:
            st.error("Please enter a project ID.")
            return

        if not training_key:
            st.error("Please enter a training key.")
            return

        # download = st.button("Download", on_click=disable, disabled=st.session_state.disabled)
        download = st.button("Download")
        if download:
            if valid_ratio:
                # Xóa toàn bộ nội dung của thư mục data trước khi tạo mới
                clear_data_folder()
                # Delete all files and folders in runs directory
                if os.path.exists('runs'):
                    # Nếu tồn tại, xóa nó
                    shutil.rmtree('runs')
                data_dir = "data/train"
                os.makedirs(data_dir, exist_ok=True)    
                images_dir = os.path.join(data_dir, "images")
                labels_dir = os.path.join(data_dir, "labels")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                downloaded_files = download_images(project_id, training_key)
                if downloaded_files:
                    split_train_valid(images_dir, labels_dir, float(valid_ratio))
                    # enable()
                    st.success("Images downloaded and split successfully!")

                    # st.rerun()

            else:
                st.error("Please enter a validation split ratio.")


    elif conversion_method == "Import from a ZIP file":
        uploaded_file = st.file_uploader("Upload ZIP file:", type=["zip"])

        if uploaded_file is not None:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("data")

            create_class_mapping("data/data.yaml", "data/mapping.txt") 

            st.success("ZIP file uploaded and extracted successfully!")

# Step 2: Training
def training():

    st.title("Step 2: Training")
    st.write("Training the model from the database and saving the weights for the object detection step")

    # epochs = 50
    # imgsz = 640
    # optimizer = "AdamW"
    # lr0 = 0.001
    # lrf = 0.1
    # patience = 60
    # batch = 16
    # dropout = 0.3

    # Input fields for training parameters
    epochs_info = "Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance."
    epochs = st.number_input("Epochs", value=100, help=epochs_info)

    patience_info = "Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus."
    patience = st.number_input("Patience", value=100, help=patience_info)

    batch_info = "Batch size for training, indicating how many images are processed before the model's internal parameters are updated. AutoBatch (batch=-1) dynamically adjusts the batch size based on GPU memory availability."
    batch = st.number_input("Batch Size", value=16, help=batch_info)

    imgsz_info = "Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity."
    imgsz = st.number_input("Image Size", value=640, help=imgsz_info)

    optimizer_info = "Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability."
    optimizer_options = ['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto']
    optimizer = st.selectbox("Optimizer", options=optimizer_options, index=optimizer_options.index('auto'), help=optimizer_info)

    lr0_info = "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated."
    lr0 = st.number_input("Initial Learning Rate", value=0.01, help=lr0_info)

    lrf_info = "Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time."
    lrf = st.number_input("Final learning rate", value=0.01, help=lrf_info)

    dropout_info = "Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training."
    dropout = st.number_input("Dropout", value=0.0, help=dropout_info)

    # if "disabled" not in st.session_state:
    #     st.session_state.disabled = False
    # training = st.button("Start Training", on_click=disable, disabled=st.session_state.disabled)
    training = st.button("Start Training")   
    # Button to start training
    if training: 
        run_yolo_training(epochs, imgsz, optimizer, lr0, lrf, patience, batch, dropout)
        # enable()    
        save_best_model()
        # st.rerun() 

# Step 3: Object Detection
def object_detection():

    st.title("Step 2: Object Detection")
    st.write("Object detection with trained models")

    # Sidebar
    st.sidebar.header("Model Config")

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 50)) / 100


    # Dropdown menu to select .pt file
    selected_pt_file = st.sidebar.selectbox("Select .pt File", list_pt_files())

    # Button to delete selected .pt file
    if selected_pt_file:
        delete_button = st.sidebar.button("Delete Selected .pt File")
        if delete_button:
            try:
                os.remove(os.path.join("weights", selected_pt_file))
                st.success(f"{selected_pt_file} deleted successfully!")
                # Update the list of .pt files after deletion
                selected_pt_file = None
            except Exception as e:
                st.error(f"Error occurred while deleting {selected_pt_file}: {str(e)}")

    # Initialize the YOLO model with the selected .pt file
    if selected_pt_file:
        model = YOLO(os.path.join("weights", selected_pt_file))
        st.sidebar.success(f"Model initialized with {selected_pt_file}")    
    
    st.sidebar.header("Image Config")

    # Image selection
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    col1, col2 = st.columns(2)

    if source_img is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize the image
        short_side = 1200
        height, width, _ = uploaded_image.shape
        if height < width:
            new_width = short_side
            new_height = int(height * (short_side / width))
        else:
            new_height = short_side
            new_width = int(width * (short_side / height))
        resized_image = cv2.resize(uploaded_image, (new_width, new_height))

        # Convert color space to RGB
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Display uploaded image in column 1
        col1.image(resized_image_rgb, caption="Uploaded Image", use_column_width=True)

        if st.sidebar.button('Detect Objects'):
            # Detect objects in the uploaded image
            results = model.predict(resized_image, iou=0.8, conf=confidence)

            # Process the results and add to prediction_dict
            for result in results:
                obj_conf = result.boxes.conf.tolist()
                class_id = result.boxes.cls.tolist()
                bbox = result.boxes.xyxy.tolist()

                for i in range(len(bbox)):
                    class_name = class_mapping.get(str(int(class_id[i])), "Unknown")
                    x1, y1, x2, y2 = bbox[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Create a new entry in prediction_dict
                    prediction_dict[i] = {
                        "class_name": class_name,  
                        "bbox": [x1, y1, x2, y2]             
                    }

                    # Draw bounding box on the original image
                    cv2.rectangle(resized_image_rgb, (x1, y1), (x2, y2), get_color(int(class_id[i])), 2)

                    label = f'{prediction_dict[i]["class_name"]}'

                    t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]

                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(resized_image_rgb, (x1, y1), c2, get_color(int(class_id[i])), -1, cv2.LINE_AA)
                    cv2.putText(resized_image_rgb, label, (x1, y1 - 2), 0, 0.6, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

            # Display detected image in column 2
            col2.image(resized_image_rgb, caption='Detected Image', use_column_width=True)

            # Display detection results in an expander
            try:
                with st.expander("Detection Results", expanded=True):
                    # Dictionary to store product counts
                    product_counts = {}
                    
                    # Loop through items in prediction_dict
                    for _, product_info in prediction_dict.items():
                        class_name = product_info["class_name"]
                        if class_name in product_counts:
                            product_counts[class_name]["Quantity"] += 1
                        else:
                            product_counts[class_name] = {
                                "Class Name": class_name,
                                "Quantity": 1
                            }

                    # Display information about products and their quantities
                    for product_info in product_counts.values():
                        st.write("Class Name:", product_info["Class Name"])
                        st.write("Quantity:", product_info["Quantity"])
                        st.write()

            except Exception as ex:
                st.write("No image is uploaded yet!")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Import Database", "Training", "Object Detection"])

    # Add a horizontal line in the sidebar
    st.sidebar.markdown("---")

    if app_mode == "Import Database":
        import_database()
    elif app_mode == "Training":
        training()
    elif app_mode == "Object Detection":
        object_detection()


    # Hide Streamlit style
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
