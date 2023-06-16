import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
# Load face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model
model = YOLO('yolov8n.pt')

# Start capturing the video file
cap = cv2.VideoCapture('test_videos/test_video_home.mp4')

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_videos/output_home.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# List to store face images and embeddings
face_embeddings = []

# Initialize last verification time to current time
last_verification_time = time.time()

# Initialize verification flag and target person's center coordinates
verified = False
center_x, center_y = 0, 0
min_distance = 50
while True:  # You want continuous detection
    # Initialize match flag
    match_found = False    
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the model to the frame
    results = model(frame)

    # Process the results
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            # single box
            box = boxes[i]
            if box.cls[0].item() == 0 and round(box.conf[0].item(), 2) >=0.75:  # 'person' with confidence >= 0.75
                # Get normalized bounding box coordinates
                x1_n, y1_n, x2_n, y2_n = box.xyxyn[0]
                x1, y1, x2, y2 = box.xyxy[0]

                # Calculate the absolute center of the bounding box
                center_x = int((x1 + x2) / 2 )
                center_y = int((y1 + y2) / 2 )

                # Crop the person from the frame
                person0 = frame[int(y1):int(y2), int(x1):int(x2)]

                # Face detection on the cropped person
                faces = face_cascade.detectMultiScale(person0, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # If a face is detected
                for (x, y, w, h) in faces:
                    # Crop the face from the person
                    face0 = person0[y:y+h, x:x+w]
                    # If there are less than 10 images, save the current image and its embedding in list
                    if len(face_embeddings) < 10:
                        try:
                            face_embeddings.append(DeepFace.represent(face0, model_name='VGG-Face', detector_backend='opencv'))
                            print("Face detected, collecting images...")
                        except ValueError:
                            continue


                # If 10 images are collected and at least 5 seconds have passed since the last verification, perform face recognition
                if len(face_embeddings) >= 10 and time.time() - last_verification_time >= 5:
                    print("10 images collected, performing face recognition...")
                    embedding = DeepFace.represent(face0, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                    distances = [np.linalg.norm(np.array(embedding[0]['embedding']) - np.array(face_embedding[0]['embedding'])) for face_embedding in face_embeddings]
                    average_distance = sum(distances) / len(distances)
                    print("AVG Distances: ", average_distance)
                    if average_distance <= min_distance:  # Threshold of 0.5, you might need to adjust it
                        print(f"Person detected at ({center_x}, {center_y})")
                        verified = True
                        match_found = True
                        last_verification_time = time.time()
                        min_distance = average_distance + 0.2
                        if min_distance < 0.2:
                            min_distance = 0.3
                            
                        break
                    else:
                        print("Person not detected")
                        verified = False
                if match_found:
                    break
                    

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Draw a green circle if person is verified, red otherwise
    if verified:
        cv2.circle(annotated_frame, (50,50), 25, (0,255,0), -1)
        
        # add a blue circle at the center of verified person
        cv2.circle(annotated_frame, (center_x, center_y), 25, (255,0,0), -1)
    else:
        cv2.circle(annotated_frame, (50,50), 25, (0,0,255), -1)

    # Write the frame into the file 'output.mp4'
    out.write(annotated_frame)

# Release the VideoCapture and VideoWriter object and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()









