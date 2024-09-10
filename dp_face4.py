import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import queue
from datetime import datetime
import pyttsx3
import csv
import ffmpeg
import subprocess

# Load the DNN face detector model from OpenCV
face_net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt',  # Path to deploy.prototxt file
    'res10_300x300_ssd_iter_140000.caffemodel'  # Path to res10_300x300_ssd_iter_140000.caffemodel
)

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

def normalize_confidence(distance, metric):
    if metric == "cosine":
        confidence = 1 - distance
    elif metric in ["euclidean", "euclidean_l2"]:
        confidence = np.exp(-distance)
    else:
        confidence = 0
    return confidence

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def mark_attendance(name):
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, time_string])

def face_recognition_thread(frame_queue, stop_event):
    backend = backends[0]

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Detect faces using OpenCV DNN module
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                    # Perform face recognition on the detected face
                    face = frame[y:y1, x:x1]
                    try:
                        people = DeepFace.find(
                            img_path=face,
                            db_path="dataset/",
                            anti_spoofing=False,  # Temporarily disable anti-spoofing for debugging
                            model_name=models[7],
                            distance_metric=metrics[0],
                            detector_backend=backend,
                            enforce_detection=False
                        )
                    except Exception as e:
                        print("Error during face recognition:", e)
                        people = []

                    for person in people:
                        if 'identity' in person and len(person['identity']) > 0:
                            name = person['identity'][0].split('/')[1]

                            # Calculate confidence from distance
                            distance = person['distance'][0]
                            confidence = normalize_confidence(distance, metrics[0])
                            text = f"{name} ({confidence:.2f})"
                            cv2.putText(frame, text, (x, y-10), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

                            # Print the result in the console
                            print(f"Identified: {name} with confidence {confidence:.2f}")

                            # Speak welcome message
                            welcome_message = f"Welcome {name}"
                            speak_text(welcome_message)

                            # Mark attendance
                            mark_attendance(name)

                            # Stop further processing once a face is recognized
                            stop_event.set()
                            return

            # Display the resulting frame
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 960, 720)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

def realtime_face_recognition():
    url = "rtsp://admin:Admin@123@192.168.102.63:554/Streaming/channels/101"  # Replace with your actual RTSP URL

    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # Replace with your actual FFmpeg path if necessary

    # Initialize FFmpeg process
    process = (
        ffmpeg
        .input(url)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True, cmd=ffmpeg_path)
    )

    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Start the face recognition thread
    threading.Thread(target=face_recognition_thread, args=(frame_queue, stop_event), daemon=True).start()

    while not stop_event.is_set():
        # Read frame from FFmpeg process
        in_bytes = process.stdout.read(640 * 480 * 3)  # Adjust the frame size if necessary
        if not in_bytes:
            print("Failed to grab frame from RTSP stream.")
            break

        # Convert bytes to numpy array
        frame = np.frombuffer(in_bytes, np.uint8).reshape((480, 640, 3))  # Adjust the frame size if necessary

        # Clear the frame queue if it is full
        if frame_queue.full():
            print("Frame queue is full, clearing old frames...")
            while not frame_queue.empty():
                frame_queue.get()

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            print("Frame queue is full, skipping frame...")

    process.stdin.close()
    process.wait()
    cv2.destroyAllWindows()

# Perform real-time face recognition using the RTSP stream
realtime_face_recognition()
