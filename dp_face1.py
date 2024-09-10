import cv2
import numpy as np
from deepface import DeepFace
import time
from ultralytics import YOLO

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your YOLOv8 weights

def normalize_confidence(distance, metric):
    if metric == "cosine":
        confidence = 1 - distance
    elif metric in ["euclidean", "euclidean_l2"]:
        # Apply a softmax-like normalization
        confidence = np.exp(-distance)
    else:
        confidence = 0
    return confidence

def realtime_face_recognition():
    # Define a video capture object
    vid = cv2.VideoCapture(0)
    backend = backends[0]  # Choose a backend from the list

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()
        start_time = time.time()

        # Perform YOLO object detection
        results = model(frame)
        phone_detected = False
        phone_center_x, phone_center_y = 0, 0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                if label == 'cell phone':  # Check for phone detection
                    xyxy = box.xyxy[0]  # Get the coordinates
                    x1, y1, x2, y2 = map(int, xyxy)
                    phone_center_x = (x1 + x2) // 2
                    phone_center_y = (y1 + y2) // 2
                    phone_detected = True
                    break

        if phone_detected:
            cv2.putText(frame, "Phone Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Alert: Phone detected, not a live image!")
            # Optionally, add text-to-speech alert here
            break

        # Perform face recognition on the captured frame
        try:
            people = DeepFace.find(
                img_path=frame,
                db_path="dataset/",
                anti_spoofing=True,
                model_name=models[7],
                distance_metric=metrics[0],
                detector_backend=backend,  # Specify the backend here
                enforce_detection=False
            )
        except Exception as e:
            print("Error during face recognition:", e)
            people = []

        end_time = time.time()
        recognition_time = end_time - start_time

        # Print the recognition time in the terminal
        print(f"Recognition Time: {recognition_time:.2f} seconds")

        for person in people:
            # Ensure that the coordinates are available
            if 'source_x' in person and len(person['source_x']) > 0:
                x = person['source_x'][0]
                y = person['source_y'][0]
                w = person['source_w'][0]
                h = person['source_h'][0]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Get the person's name and display it on the image
                if 'identity' in person and len(person['identity']) > 0:
                    name = person['identity'][0].split('/')[1]

                    # Calculate confidence from distance
                    distance = person['distance'][0]
                    confidence = normalize_confidence(distance, metrics[0])
                    text = f"{name} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        # Check if the 'q' button is pressed to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    vid.release()
    cv2.destroyAllWindows()

# Perform real-time face recognition using the webcam
realtime_face_recognition()