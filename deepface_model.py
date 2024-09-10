import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Path to the dataset
dataset_path = 'dataset/'

# Path to store attendance logs
attendance_path = 'attendance.csv'

# Function to encode and store face embeddings
def encode_faces(dataset_path):
    employees = {}
    for employee_dir in os.listdir(dataset_path):
        employee_path = os.path.join(dataset_path, employee_dir)
        if os.path.isdir(employee_path):
            employees[employee_dir] = []
            for filename in os.listdir(employee_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    file_path = os.path.join(employee_path, filename)
                    embedding = DeepFace.represent(file_path, model_name='DeepFace')[0]['embedding']
                    employees[employee_dir].append(np.array(embedding))
    return employees

# Function to recognize a face from an input image
def recognize_face(image_path, employees, threshold=0.5):
    try:
        captured_embedding = DeepFace.represent(image_path, model_name='DeepFace')[0]['embedding']
        captured_embedding = np.array(captured_embedding)
        
        recognized_employee_id = None
        best_accuracy = float('inf')

        for employee_id, embeddings in employees.items():
            for embedding in embeddings:
                distance = np.linalg.norm(captured_embedding - embedding)
                if distance < best_accuracy:
                    best_accuracy = distance
                    recognized_employee_id = employee_id
        
        if best_accuracy < threshold:
            accuracy = 1 - best_accuracy
            return recognized_employee_id, accuracy
        else:
            return None, None
    except Exception as e:
        print(f"Error recognizing face: {e}")
        return None, None

# Function to log attendance
def log_attendance(employee_id, attendance_path):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance_data = pd.DataFrame({'EmployeeID': [employee_id], 'Timestamp': [current_time]})
    if not os.path.isfile(attendance_path):
        attendance_data.to_csv(attendance_path, index=False)
    else:
        attendance_data.to_csv(attendance_path, mode='a', header=False, index=False)
    print(f"Attendance logged for employee {employee_id} at {current_time}")

# Function to capture image from webcam
def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press "q" to capture image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    img_path = 'captured_image.jpg'
    cv2.imwrite(img_path, frame)
    return img_path

# Main function
def main():
    employees = encode_faces(dataset_path)
    img_path = capture_image_from_webcam()
    
    employee_id, accuracy = recognize_face(img_path, employees)
    if employee_id:
        log_attendance(employee_id, attendance_path)
        print(f"Employee ID: {employee_id}, Accuracy: {accuracy:.2f}")
    else:
        print("Employee not recognized.")

if __name__ == "__main__":
    main()
