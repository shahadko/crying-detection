import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# Path to your YOLO model and test images folder
model_path = r"C:\Users\shahad\Downloads\model.pt"
test_images_path = r"C:\Users\shahad\Documents\testimage"

# Load YOLO model
model = YOLO(model_path)

# Load pre-trained face and feature detectors from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

# Create 'result' directory if it doesn't exist
result_folder = os.path.join(test_images_path, 'result')
os.makedirs(result_folder, exist_ok=True)

# Detect crying based on mouth and squinted eyes
def detect_crying_baby(img, face_region):
    # Convert the face region to grayscale for feature detection
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Detect mouth in the face region
    mouths = mouth_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    status = "NOT CRYING"
    
    # Check if mouth is open (potential crying) and eyes are squinted (often associated with crying)
    if len(mouths) > 0 and len(eyes) < 2:  # If less than two eyes are detected, crying is more likely
        for (mx, my, mw, mh) in mouths:
            if mh > 30:  # Adjust the mouth height threshold to a more sensitive value
                status = "CRYING"
                break  # If mouth is open, we stop further checks
    
    return status

# Process images in the folder
for img_file in Path(test_images_path).glob('*.*'):  # Iterates over all image files
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        print(f"Processing image: {img_file}")
        
        # Read the image
        img = cv2.imread(str(img_file))
        
        # Perform object detection using YOLO
        results = model(img)
        
        # Initialize status
        status = "UNKNOWN"
        
        # Extract detected class names (e.g., "open" or "closed")
        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
        
        if "open" in detected_classes:
            status = "AWAKE"
        elif "closed" in detected_classes:
            status = "ASLEEP"
        
        # Detect faces in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Check for crying based on the detected faces
        if len(faces) > 0:
            status = "BABY DETECTED"
            for (x, y, w, h) in faces:
                # Crop the detected face region for emotion detection
                face_region = img[y:y+h, x:x+w]
                
                # Check if the baby is crying based on mouth and eyes
                crying_status = detect_crying_baby(img, face_region)
                if crying_status == "CRYING":
                    status = "CRYING"
                    break  # No need to check further faces once crying is detected
        
        # Print the final status in terminal
        print(f"Status: {status}")
        
        # Annotate the image with the detection results (without text on image)
        annotated_frame = results[0].plot()  # Get YOLO annotations
        
        # Add text to indicate the baby is crying, awake, or asleep
        if status == "CRYING":
            cv2.putText(annotated_frame, "Crying", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif status == "AWAKE":
            cv2.putText(annotated_frame, "Awake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif status == "ASLEEP":
            cv2.putText(annotated_frame, "Asleep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Save the annotated image to the result folder
        result_image_path = os.path.join(result_folder, img_file.name)
        cv2.imwrite(result_image_path, annotated_frame)
        print(f"Saved result image to: {result_image_path}")
