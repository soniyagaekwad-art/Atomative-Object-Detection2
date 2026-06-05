import streamlit as st
import cv2
import numpy as np
import easyocr
import tempfile
import os
import time
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle & Plate Detector", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .main-title { font-size:2.5rem; font-weight:800; color:#00d4ff; text-align:center; margin-bottom:0.2rem; }
    .sub-title { text-align:center; color:#aaaaaa; font-size:1rem; margin-bottom:2rem; }
    .detection-card { background:#1e2130; border-radius:10px; padding:12px 16px; margin-bottom:8px; border-left:4px solid #00d4ff; }
    .vehicle-tag { font-weight:bold; color:#00d4ff; font-size:1.1rem; }
    .plate-tag { color:#f0c040; font-size:1rem; font-weight:bold; }
    .confidence-tag { color:#aaaaaa; font-size:0.85rem; }
    .stButton > button { background:linear-gradient(135deg,#00d4ff,#0066ff); color:white; border:none; border-radius:8px; font-weight:bold; padding:0.5rem 2rem; width:100%; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 Vehicle & Number Plate Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a traffic video — detect vehicle type & license plate using YOLOv8 + EasyOCR</div>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

VEHICLE_CLASSES = {2:"Car",3:"Motorcycle",5:"Bus",7:"Truck",1:"Bicycle"}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())
BOX_COLORS = {2:(0,212,255),3:(0,255,128),5:(255,100,0),7:(0,80,255),1:(200,0,255)}

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def read_plate(ocr_reader, plate_img):
    try:
        preprocessed = preprocess_plate(plate_img)
        results = ocr_reader.readtext(preprocessed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if results:
            text = "".join([r[1] for r in results if r[2] > 0.2]).strip().upper()
            return text if len(text) >= 4 else "N/A"
        return "N/A"
    except Exception:
        return "N/A"

def draw_label(frame, text, x, y, color, bg=(20,20,20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    cv2.rectangle(frame, (x-pad, y-th-pad*2), (x+tw+pad, y+baseline), bg, -1)
    cv2.putText(frame, text, (x, y-pad), font, scale, color, thickness, cv2.LINE_AA)

def process_frame(frame, yolo_model, ocr_reader, conf_threshold):
    detections = []
    results = yolo_model(frame, conf=conf_threshold, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASS_IDS:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = VEHICLE_CLASSES[cls_id]
        color = BOX_COLORS.get(cls_id, (255,255,255))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        draw_label(frame, f"{label} {conf:.0%}", x1, y1, color)
        plate_y1 = y1 + int((y2-y1)*0.6)
        plate_crop = frame[plate_y1:y2, x1:x2]
        plate_text = "N/A"
        if plate_crop.size > 0 and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
            plate_text = read_plate(ocr_reader, plate_crop)
        if plate_text != "N/A":
            draw_label(frame, f"Plate:{plate_text}", x1, y2+22, (240,192,0), (40,30,0))
        detections.append({"vehicle":label,"plate":plate_text,"confidence":f"{conf:.1%}"})
    return frame, detections

# SIDEBAR
with st.sidebar:
    st.markdown("## Settings")
    conf_thresh = st.slider("Detection Confidence", 0.25, 0.95, 0.45, 0.05)
    frame_skip  = st.slider("Process Every N Frames", 1, 10, 3)
    max_frames  = st.number_input("Max Frames to Process (0 = all)", 0, 5000, 300)
    st.markdown("---")
    st.markdown("**Vehicle Types:** Car | Motorcycle | Bus | Truck | Bicycle")
    st.markdown("**Models:** YOLOv8n + EasyOCR")

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4","avi","mov","mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()
    video_path = tfile.name

    st.success(f"Uploaded: {uploaded_file.name}")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        start_btn = st.button("Start Detection", use_container_width=True)

    if start_btn:
        with st.spinner("Loading YOLOv8..."):
            yolo = load_yolo()
        with st.spinner("Loading EasyOCR..."):
            ocr = load_ocr()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video.")
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        limit = int(max_frames) if max_frames > 0 else total_frames

        st.markdown("---")
        left_col, right_col = st.columns([3,2])
        with left_col:
            st.markdown("### Live Detection Feed")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text  = st.empty()
        with right_col:
            st.markdown("### Detection Log")
            log_placeholder = st.empty()

        all_detections = []
        frame_count    = 0
        processed      = 0
        detection_log  = []

        while cap.isOpened() and frame_count < limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            progress_bar.progress(min(frame_count / limit, 1.0))
            status_text.markdown(f"Processing frame **{frame_count}/{limit}**")

            if frame_count % frame_skip != 0:
                continue

            processed += 1
            annotated, dets = process_frame(frame, yolo, ocr, conf_thresh)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            for d in dets:
                all_detections.append(d)
                detection_log.insert(0, d)
                if len(detection_log) > 50:
                    detection_log.pop()

            log_html = ""
            seen = set()
            for d in detection_log[:15]:
                key = f"{d['vehicle']}-{d['plate']}"
                if key in seen:
                    continue
                seen.add(key)
                log_html += f"""<div class="detection-card">
                    <span class="vehicle-tag">{d['vehicle']}</span><br>
                    <span class="plate-tag">Plate: {d['plate']}</span><br>
                    <span class="confidence-tag">Confidence: {d['confidence']}</span>
                </div>"""
            log_placeholder.markdown(log_html, unsafe_allow_html=True)

        cap.release()
        progress_bar.progress(1.0)
        status_text.markdown(f"Done! Processed **{processed}** frames out of **{frame_count}**.")

        st.markdown("---")
        st.markdown("## Detection Summary")

        if all_detections:
            total  = len(all_detections)
            plates = [d for d in all_detections if d["plate"] != "N/A"]
            types  = {}
            for d in all_detections:
                types[d["vehicle"]] = types.get(d["vehicle"], 0) + 1

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Detections", total)
            m2.metric("Plates Read", len(plates))
            m3.metric("Vehicle Types", len(types))

            st.markdown("### Vehicle Type Breakdown")
            for vtype, count in sorted(types.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                st.markdown(f"**{vtype}** — {count} ({pct:.1f}%)")
                st.progress(count / total)

            if plates:
                st.markdown("### All Detected Plates")
                unique_plates = list({d["plate"] for d in plates})
                st.code("\n".join(unique_plates), language="text")
        else:
            st.warning("No vehicles detected. Try lowering the confidence threshold.")

        try:
            os.remove(video_path)
        except Exception:
            pass

else:
    st.markdown("""
    <div style='text-align:center;padding:3rem;background:#1e2130;border-radius:12px;border:2px dashed #334;'>
        <div style='font-size:3rem'>🎬</div>
        <div style='color:#aaa;font-size:1.1rem;margin-top:1rem'>
            Upload a <strong style='color:#00d4ff'>traffic video</strong> above to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
