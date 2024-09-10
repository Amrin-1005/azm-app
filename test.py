# import cv2

# def test_rtsp_stream(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         print("Failed to open RTSP stream")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         cv2.imshow('RTSP Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Replace with your actual RTSP URL
# rtsp_url = "rtsp://admin:Admin@123@192.168.102.63:554/Streaming/channels/101"
# test_rtsp_stream(rtsp_url)

# import cv2
# import numpy as np
# import threading
# import queue
# import time

# # Load the DNN face detector model from OpenCV
# face_net = cv2.dnn.readNetFromCaffe(
#     'deploy.prototxt.txt',  # Path to deploy.prototxt file
#     'res10_300x300_ssd_iter_140000.caffemodel'  # Path to res10_300x300_ssd_iter_140000.caffemodel
# )

# def face_detection_thread(frame_queue, stop_event):
#     while not stop_event.is_set():
#         try:
#             frame = frame_queue.get(timeout=5)  # Wait up to 1 second for a frame

#             (h, w) = frame.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#             face_net.setInput(blob)
#             detections = face_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
#                 if confidence > 0.5:
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (x, y, x1, y1) = box.astype("int")
#                     cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

#             cv2.imshow('Frame', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_event.set()
#                 break

#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"Error in face detection thread: {e}")
#             stop_event.set()
#             break

# def realtime_video_capture():
#     rtsp_url = "a.mp4"  # Replace with your actual RTSP URL
#     vid = cv2.VideoCapture(rtsp_url)
#     vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
#     vid.set(cv2.CAP_PROP_FPS, 10)  # Set frame rate to 10 FPS

#     if not vid.isOpened():
#         print("Failed to open video stream at", rtsp_url)
#         return

#     frame_queue = queue.Queue(maxsize=5)
#     stop_event = threading.Event()

#     detection_thread = threading.Thread(target=face_detection_thread, args=(frame_queue, stop_event), daemon=True)
#     detection_thread.start()

#     while not stop_event.is_set():
#         ret, frame = vid.read()

#         if not ret:
#             print("Failed to grab frame from CCTV camera.")
#             stop_event.set()
#             break

#         if not frame_queue.full():
#             frame_queue.put(frame)
#         else:
#             print("Frame queue is full, skipping frame...")

#         time.sleep(0.5)  # Add a small delay to reduce frame capture rate

#     vid.release()
#     cv2.destroyAllWindows()
#     detection_thread.join()

# realtime_video_capture()
import cv2
import time
import dlib
from deepface import DeepFace

# Initialize the DeepFace model
model_name = "Facenet512"  # Use a faster model
db_path = "dataset/"
detector = dlib.get_frontal_face_detector()

def recognize_face(face_image):
    try:
        results = DeepFace.find(img_path=face_image, db_path=db_path, model_name=model_name, enforce_detection=False)
        if len(results) > 0:
            return results[0]['identity'][0].split('/')[-1]
        else:
            return "Unknown"
    except Exception as e:
        print("Error during face recognition:", e)
        return "Unknown"

def open_rtsp_camera(rtsp_url):
    max_retries = 5
    retry_delay = 5
    reconnect_delay = 10
    cap = None
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 50)

            if not cap.isOpened():
                print("Error: Unable to open the RTSP stream.")
                time.sleep(reconnect_delay)
                continue

            print("RTSP stream opened successfully.")
        
        retries = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame from the RTSP stream.")
                retries += 1
                if retries >= max_retries:
                    print("Max retries reached. Reconnecting...")
                    cap.release()
                    cap = None
                    break
                time.sleep(retry_delay)
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            retries = 0

            try:
                # Resize the frame for faster processing
                small_frame = cv2.resize(frame, (320, 240))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect faces using dlib
                faces = detector(rgb_frame)

                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    # Extract the face from the frame
                    face_image = rgb_frame[y:y+h, x:x+w]
                    face_image_path = 'temp_face.jpg'
                    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                    # Recognize the face
                    name = recognize_face(face_image_path)
                    print(f"Detected face: {name}")

                    # Draw a rectangle around the detected face and label it
                    cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(small_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('RTSP Stream', small_frame)

            except Exception as e:
                print(f"Error during frame processing: {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        time.sleep(reconnect_delay)

# Replace with your actual RTSP URL
rtsp_url = "rtsp://admin:Admin@123@192.168.102.63:554/Streaming/channels/101"
open_rtsp_camera(rtsp_url)






