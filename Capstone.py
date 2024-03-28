import os
import cv2
import time

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Directory to save screenshots
save_dir = 'screenshots'

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Counter for naming the screenshots
screenshot_count = 0

# Timestamp of the last screenshot
last_screenshot_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save the face as a screenshot every 3 seconds
        current_time = time.time()
        if current_time - last_screenshot_time >= 3:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_dir, f'face1_{screenshot_count}.jpg'), face_img)
            screenshot_count += 1
            last_screenshot_time = current_time
    
    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

