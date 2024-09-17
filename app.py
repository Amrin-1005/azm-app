# face recognition with sql working fine

import math
from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import base64
from flask_cors import CORS
import logging
import traceback
import os
from datetime import datetime
import pyodbc
import threading

app = Flask(__name__)
CORS(app)

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

DB_CONFIG = {
    'server': 'azmapp.database.windows.net',
    'database': 'AzmDB',
    'username': 'azm',
    'password': '@Amrin1005',
    'driver': '{ODBC Driver 17 for SQL Server}',
}

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the DeepFace models
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

def get_db_connection():
    """Create and return a new database connection using global DB_CONFIG."""
    try:
        conn = pyodbc.connect(
            f'DRIVER={DB_CONFIG["driver"]};'
            f'SERVER={DB_CONFIG["server"]};'
            f'PORT=1433;'
            f'DATABASE={DB_CONFIG["database"]};'
            f'UID={DB_CONFIG["username"]};'
            f'PWD={DB_CONFIG["password"]}'
        )
        return conn
    except Exception as e:
        logging.error("Error connecting to the database: %s", str(e))
        raise

def insert_attendance(employee_id, timestamp):
    try:
        # Establish a connection to Azure SQL
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL query to insert data
        query = "INSERT INTO OSEmployeeInfo (EmployeeCode, Timestamp) VALUES (?, ?)"
        cursor.execute(query, (employee_id, timestamp))
        
        # Commit the transaction
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Attendance record inserted successfully")
    except Exception as e:
        logging.error("Error inserting attendance record: %s", str(e))
        

def process_and_insert(identity_list, current_time):
    """Function to process data and insert it into SQL in a separate thread."""
    if identity_list:
        # Insert the first match into Azure SQL
        insert_attendance(identity_list[0], current_time)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        data = request.json
        image_data = base64.b64decode(data['image'])
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Perform face recognition directly on the image
        backend = backends[0]  # Choose a backend
        results = DeepFace.find(
            img_path=img,
            db_path="images_OS",
            anti_spoofing=False,  # Adjust as needed
            model_name="Dlib",
            distance_metric="cosine",
            detector_backend="opencv",
            enforce_detection=False
        )

        # Log and check the result
        logging.info("Returned data: %s", results)
        if results and isinstance(results, list) and len(results) > 0:
            df = results[0]
            if not df.empty:
                # Extract and trim the identity column directly
                identity_list = df['identity'].apply(lambda x: os.path.splitext(os.path.basename(x))[0].replace('_', ' ')).tolist()
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(current_time)
                # Start a new thread for inserting attendance data
                threading.Thread(target=process_and_insert, args=(identity_list, current_time)).start()

                # Immediately respond to the client with the recognition result
                return jsonify({"matches": identity_list, "timestamp": current_time})

                
            else:
                return jsonify({"error": "No match found"}), 404
        else:
            return jsonify({"error": "No match found"}), 404

    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Traceback: %s", traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500
    
    
def get_nearby_geofences(user_lat, user_lon):
    print('Received request')
    print(user_lat,user_lon)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = '''
    SELECT ItemID, name, latitude, longitude, radius
    FROM Geofences
    '''
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    geofences = []
    for row in rows:
        geofence_lat = row.latitude
        geofence_lon = row.longitude
        
        # Calculate distance
        distance = haversine_distance(user_lat, user_lon, geofence_lat, geofence_lon)
        
        # Check if the distance is within 100 meters
        if distance <= 100:
            geofence = {
                'ItemID': row.ItemID,
                'name': row.name,
                'latitude': row.latitude,
                'longitude': row.longitude,
                'radius': row.radius
            }
            geofences.append(geofence)
            for item in geofence:
                print(item)
    
    cursor.close()
    conn.close()
    
    return geofences

@app.route('/geofences', methods=['POST'])
def get_nearby_geofences_endpoint():
    data = request.json
    user_lat = data['latitude']
    user_lon = data['longitude']
    
    print(f"Received latitude: {user_lat}, longitude: {user_lon}")
    
    if user_lat is None or user_lon is None:
        return jsonify({"error": "Latitude or longitude not provided"}), 400
    
    nearby_geofences = get_nearby_geofences(user_lat, user_lon)
    
    return jsonify({'geofences': nearby_geofences})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
#     app.debug = True
