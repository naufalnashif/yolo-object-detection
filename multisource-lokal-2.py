import cv2
import threading
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Daftar sumber video
camera_sources = [
    'video-inputs/parking-detect.mp4',
    'video-inputs/vehicle-count-1.mp4',
    'video-inputs/people-walking.mp4',
    'video-inputs/vehicle-count-3.mp4'
]

SOURCE_OUTPUT_PATH = "outputs/video-output.mp4"
RESULT_CSV_FILE_PATH = "outputs/detection.csv"
color_hexes = ['#2190ff', '#6666FF', '#FF6666', '#ff6f21', '#ffc021', '#66FFFF']
color_palette = sv.ColorPalette.from_hex(color_hexes)

# Variabel global untuk menyimpan frame yang akan ditampilkan dari setiap sumber video
display_frames = {source: None for source in camera_sources}

# Fungsi untuk memproses setiap sumber video
def process_video(source):
    cap = cv2.VideoCapture(source)
    model = YOLO('yolov8n.pt')
    tracker = sv.ByteTrack(minimum_matching_threshold=0.8)  # Misalnya Anda punya tracker
    
    box_annotator = sv.BoundingBoxAnnotator(color=color_palette, thickness=2)
    label_annotator = sv.LabelAnnotator(color=color_palette, text_padding=5)
    trace_annotator = sv.TraceAnnotator(color=color_palette, thickness=1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, agnostic_nms=True, conf= 0.1)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#ID:{tracker_id} {results.names[class_id]} {confidence:.2f}"
            for tracker_id, confidence, class_id
            in zip(detections.tracker_id, detections.confidence, detections.class_id)
        ]

        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

        # Simpan frame yang sudah dianotasi ke dalam variabel global display_frames
        display_frames[source] = annotated_frame

    cap.release()

# Fungsi untuk menampilkan frame dari semua sumber video dalam satu layar terbagi
def display_videos():
    # Mengatur dimensi untuk setiap sumber video
    dimensions = {
        'video-inputs/parking-detect.mp4': (0, 0),
        'video-inputs/vehicle-count-1.mp4': (0, 960),  # Lebar lebih dari 640 untuk horizontal padding
        'video-inputs/people-walking.mp4': (540, 0),  # Tinggi lebih dari 480 untuk vertical padding
        'video-inputs/vehicle-count-3.mp4': (540, 960)     # Lebar lebih dari 640 untuk horizontal padding
    }

    # Membuat layar kosong untuk menampilkan video
    combined_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Lebar dan tinggi layar lebih besar

    while True:
        # Mengisi layar gabungan dengan frame dari setiap sumber video
        for source, frame in display_frames.items():
            if frame is not None:
                y, x = dimensions[source]
                h, w = frame.shape[:2]

                # Hitung ukuran baru yang mempertahankan rasio aspek
                scale_w = 960 / w
                scale_h = 540 / h
                scale = min(scale_w, scale_h)

                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_frame = cv2.resize(frame, (new_w, new_h))

                # Hitung padding untuk menempatkan frame di tengah
                pad_x = (960 - new_w) // 2
                pad_y = (540 - new_h) // 2

                # Tempatkan frame pada layar gabungan dengan padding
                combined_frame[y+pad_y:y+pad_y+new_h, x+pad_x:x+pad_x+new_w] = resized_frame

        # Menampilkan layar gabungan di thread utama
        cv2.imshow("yolov8 - Combined", combined_frame)

        # Exit loop jika tombol ESC ditekan
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()

# Membuat dan memulai thread untuk setiap sumber video
threads = []
for source in camera_sources:
    thread = threading.Thread(target=process_video, args=(source,))
    threads.append(thread)
    thread.start()

# Menjalankan fungsi display_videos di thread utama
display_videos()

# Menunggu semua thread selesai
for thread in threads:
    thread.join()