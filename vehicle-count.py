import streamlit as st
import pandas as pd
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
# import os
from datetime import datetime
import pytz
import os
import json

# Tentukan folder yang berisi video
folder_path = "video-inputs/"


# Dapatkan daftar semua file model dalam folder
ml_models = [m for m in os.listdir("ml-models/") if m.endswith(('.pt'))]

# Dapatkan daftar semua file video dalam folder
video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]

# Buat full path untuk setiap file
video_files_full_path = [os.path.join(folder_path, f) for f in video_files]


utc_timezone = pytz.timezone('UTC')
datetime_utc = datetime.now(utc_timezone)
wib_timezone = pytz.timezone('Asia/Jakarta')
dateNow = datetime_utc.astimezone(wib_timezone)

# dateNow = datetime.now(timezone.utc)
dateSimple = dateNow.strftime("%A, %d %b %Y")
timeNow = dateNow.strftime("%H:%M WIB")
yearNow = dateNow.strftime("%Y")


SOURCE_OUTPUT_PATH = "outputs/video-output.mp4"
RESULT_CSV_FILE_PATH = "outputs/detection.csv"
color_hexes = ['#2190ff', '#6666FF', '#FF6666', '#ff6f21', '#ffc021', '#66FFFF']
color_palette = sv.ColorPalette.from_hex(color_hexes)

# Initialize annotators and other utilities
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator(color=color_palette, thickness=2)
label_annotator = sv.LabelAnnotator(color=color_palette, text_padding=5)
trace_annotator = sv.TraceAnnotator(color=color_palette, thickness=1)


# Fungsi untuk memuat dan mengonversi data dari file JSON ke array NumPy
def load_polygons_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    polygons = [np.array(polygon) for polygon in data]
    return polygons

# Path ke file JSON hasil anotasi
json_path = 'outputs/polygons.json'

# Memuat data poligon dari file JSON
polygons = load_polygons_from_json(json_path)
polygon = np.array(polygons[0])

line_zone = sv.PolygonZone(polygon=polygon)
line_annotator = sv.PolygonZoneAnnotator(line_zone, color=sv.Color.WHITE)


# Function to process video frames and save outputs
def main(source_path: str, target_path: str, callback, customClass):

    video_info = sv.VideoInfo.from_video_path(source_path)
    total_frames = video_info.total_frames

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    

    with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
        for frame_index, frame in enumerate(sv.get_video_frames_generator(source_path=source_path)):
            result_frame = callback(frame, frame_index, customClass)
            sink.write_frame(frame=result_frame)

            #Realtime video update with RGB color
            result_frame_real = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(result_frame_real)

            # Update progress bar and status text
            progress_percent = (frame_index + 1) / total_frames
            progress_bar.progress(progress_percent)
            status_text.text(f'Processing... Frame {frame_index + 1} of {total_frames}')

    st.success("✅ Processing complete")
    progress_bar.empty()
    status_text.empty()
    frame_placeholder.empty()

# Callback function for frame processing
def callback(frame: np.ndarray, frame_index: int, customClass) -> np.ndarray:
    results = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = detections[np.isin(detections.class_id, customClass)]
    

    detections = detections[line_zone.trigger(detections=detections)]


    labels = [
        f"#ID:{tracker_id} {results.names[class_id]} {confidence:.2f}"
        for tracker_id, confidence, class_id
        in zip(detections.tracker_id, detections.confidence, detections.class_id)
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)


    annotated_frame = line_annotator.annotate(annotated_frame)

    # Write detections to CSV
    write_to_csv(frame_index, detections)

    return trace_annotator.annotate(annotated_frame, detections=detections)

# Function to write detections to CSV
def write_to_csv(frame_index, detections):
    if frame_index == 0:
        with open(RESULT_CSV_FILE_PATH, 'w') as f:
            header = ['frame_index', 'tracker_id', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'confidence', 'class_name']
            f.write(','.join(header) + '\n')
    with open(RESULT_CSV_FILE_PATH, 'a') as f:
        for i in range(len(detections)):
            line = [
                frame_index,
                detections.tracker_id[i],
                detections.xyxy[i][0],
                detections.xyxy[i][1],
                detections.xyxy[i][2],
                detections.xyxy[i][3],
                detections.class_id[i],
                detections.confidence[i],
                model.model.names[detections.class_id[i]]
            ]
            f.write(','.join(map(str, line)) + '\n')

# Function to update analysis data
def update_analysis_data():
    df = pd.read_csv(RESULT_CSV_FILE_PATH)
    max_frame_index = df['frame_index'].max()

    for class_id, label in model.model.names.items():
        class_detections = df[df['class_id'] == class_id]
        unique_ids = set(class_detections['tracker_id'])
        unique_ids_per_class[label].update(unique_ids)
        
        frame_counts = class_detections.groupby('frame_index').size()
        counts_list = [frame_counts.get(i, 0) for i in range(max_frame_index + 1)]
        count_over_time[label] = counts_list

# Function to handle download button clicks
def download_button(label, file_path, key):
    with open(file_path, "rb") as file:
        file_content = file.read()
    st.download_button(label, file_content, file_path, key=key)



# ------------------------------------------UI-------------------------------------------
st.title("Video Analytics: Vehicle Count Analysis")
with st.sidebar:
    with st.container():
        st.image('assets/self-daily-logo.jpeg')
        st.text(f"Today\t: {dateSimple}")
        st.text(f"Time\t: {timeNow}")
    with st.expander("General Settings:"):
        MODEL_PATH = st.selectbox("Choose Model :", ml_models, index=0)
        if not MODEL_PATH:
            st.warning("⚠️ No model selected, using ppe-detection.pt model.")
            MODEL_PATH = "ppe-detection.pt"

        model = YOLO(MODEL_PATH)
        labels = model.model.names
        keyList = list(labels.keys())

        
        # SOURCE_VIDEO_PATH = st.selectbox("Choose video:", ("video-inputs/short-video-test.mp4", "video-inputs/720-video.mp4", "video-inputs/parking-detect.mp4"), index=0)
        # Buat selectbox untuk memilih video
        SOURCE_VIDEO_PATH = st.selectbox("Choose video:", video_files_full_path, index=0)
            
        if not SOURCE_VIDEO_PATH:
            st.warning("⚠️ No video selected, using short-video-test.mp4 model.")
            SOURCE_VIDEO_PATH = "short-video-test.mp4"


        if SOURCE_VIDEO_PATH:
            video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
            st.info(f"""
                **Video Information:**
                - **Width:** {video_info.width} pixels
                - **Height:** {video_info.height} pixels
                - **FPS:** {video_info.fps}
                - **Total Frames:** {video_info.total_frames}
            """)

    # Buat kamus terbalik untuk memetakan nilai ke kunci
    value_to_key = {v: k for k, v in labels.items()}

    # Ambil dua kelas teratas sebagai nilai default
    default_classes = list(value_to_key.keys())[:6]

    # Tampilkan multiselect dengan nilai (bukan kunci)
    with st.expander("Filter Detection:"):
        customClass_values = st.multiselect(
            "Choose Target Classes:",
            list(value_to_key.keys()),  # Daftar nilai
            default_classes  # Nilai default
        )

    # Konversi nilai yang dipilih menjadi kunci yang sesuai
    customClass = [value_to_key[value] for value in customClass_values]
    
    # Process button
    button = st.button("Process")



# ------------------------------------------PROCESS-------------------------------------------
if MODEL_PATH and SOURCE_VIDEO_PATH and button:
    # Initialize data storage for analysis
    unique_ids_per_class = {label: set() for label in model.model.names.values()}
    count_over_time = {label: [] for label in model.model.names.values()}

    # Process video
    with st.spinner('Processing video, please wait...'):
        main(source_path=SOURCE_VIDEO_PATH, target_path=SOURCE_OUTPUT_PATH, callback=callback, customClass=customClass)
    
    # Display processed video
    st.video(SOURCE_OUTPUT_PATH)

    # Update analysis data after processing
    update_analysis_data()

    # Filter data for selected classes
    filtered_unique_ids_per_class = {label: ids for label, ids in unique_ids_per_class.items() if label in customClass_values}
    unique_id_counts = {label: len(ids) for label, ids in filtered_unique_ids_per_class.items()}
    unique_id_chart = pd.DataFrame.from_dict(unique_id_counts, orient='index', columns=['Unique ID Count'])
    
    filtered_count_over_time = {label: count for label, count in count_over_time.items() if label in customClass_values}
    max_length = max(len(v) for v in filtered_count_over_time.values())
    for k in filtered_count_over_time:
        filtered_count_over_time[k] += [0] * (max_length - len(filtered_count_over_time[k]))
    
    count_over_time_chart = pd.DataFrame.from_dict(filtered_count_over_time)

    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Unique ID Counts per Class**")
        st.bar_chart(unique_id_chart)

    st.markdown("Download processed files:")

    colA, colB = st.columns(2)
    # Download button for video
    with colA:
        download_button("Download Video", "outputs/video-output.mp4", "video")

    # Download button for CSV
    with colB:
        download_button("Download CSV", "outputs/detection.csv", "csv")

# Garis pemisah
st.divider()
st.write('Thank you for trying the demo!') 
st.caption(f'Made with ❤️ by :blue[Naufal Nashif] ©️ {yearNow}')