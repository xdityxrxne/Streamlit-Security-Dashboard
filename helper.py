from ultralytics import YOLO
import streamlit as st
import cv2
from collections import defaultdict
import numpy as np
import time
import csv
import time
from datetime import datetime
import settings
import threading

count = 0

def check_camera_indices(cam_threshold):
    cam = []
    for i in range(int(cam_threshold)):  # Try indices 0 to 9 (you can adjust this range)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cam.append(i)
            cap.release()
        else:
            cap.release()
    return cam

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def index_display_tracker_options(index):
    display_tracker_key = f"tracker_radio_{index}"
    display_tracker = st.radio(f"Display Tracker {index}", ('Yes', 'No'), key=display_tracker_key)
    is_display_tracker = True if display_tracker == 'Yes' else False
    
    tracker_type_key = f"tracker_type_{index}"
    tracker_type = st.radio(f"Tracker {index}", ("bytetrack.yaml", "botsort.yaml"), key=tracker_type_key)
    
    return is_display_tracker, tracker_type


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    global count
    count = 0
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
    boxes = res[0].boxes.xywh.cpu()
    classes = np.array(res[0].boxes.cls.cpu(), dtype="int")
    print(classes)
    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    for cls,box in zip(classes,boxes):
            class_name = res[0].names[cls]
            if class_name=="person":
                count = count+1
                print(count)
            
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

 
# Define a function to write count and timestamp to a CSV file
def write_to_csv(count, csv_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, count])
        
def play_webcam(confidence, model, cam_threshold, crowd_threshold, csv_file):
    """
    Plays two webcam streams simultaneously. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        cam_threshold: Threshold for selecting cameras.
        crowd_threshold: Threshold for crowd size.
        csv_file: Path to the CSV file to save count and timestamps.

    Returns:
        None

    Raises:
        None
    """
    cam_list = check_camera_indices(cam_threshold)
    webcams = {}
    for i, camera_index in enumerate(cam_list):
        webcams[f'WEBCAM_{i}'] = camera_index
    
    vid_caps = []  
    st_frames = []  
    ts = [] 
    ws = [] 
    
    for j in range(len(webcams)):       
        is_display_tracker, tracker = index_display_tracker_options(j)
        
        if st.sidebar.button(f'Start Camera {j}'):
            try:
                vid_cap = cv2.VideoCapture(webcams[f"WEBCAM_{j}"])
                vid_caps.append(vid_cap)
                
                st_frame = st.empty()
                st_frames.append(st_frame)
                t = st.empty()
                w = st.empty()
                ws.append(w)
                ts.append(t)
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    
                    if success:
                        # Detect objects and count them (you'll need to implement this)
                        #count = detect_and_count_objects(model, image, conf)
                        
                        t.markdown(f"Number of Persons: {count}")
                        write_to_csv(count, csv_file)  # Write count and timestamp to CSV
                        
                        if count > crowd_threshold:
                            w.warning(f"🚨 Alert: Crowd size exceeds the set threshold ({crowd_threshold}) members 🚨")
                        else:
                            w.success("✅ Crowd is under control! ✅")
                        
                        # Display frames from both cameras
                        _display_detected_frames(confidence, model, st_frame, image, is_display_tracker, tracker)
                    else:
                        # Release video captures if either camera is done
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
    
    csv_file = 'count_data.csv'  # Replace with your desired CSV file path

def play_stored_video(conf, model,crowd_threshold):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            t=st.empty()
            w=st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    t.markdown(f"Number of Persons:{count}")
                    if count > crowd_threshold:
                        w.warning(f"🚨 Alert: Crowd size exceeds the set threshold ({round(crowd_threshold)}) members 🚨")
                    else:
                        w.success("✅ Crowd is under control! ✅")
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                    
                else:
                    vid_cap.release()
                    break

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))