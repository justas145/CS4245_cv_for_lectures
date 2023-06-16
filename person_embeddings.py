import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# Load model
model = YOLO('yolov8n.pt')

# Start capturing the video file
cap = cv2.VideoCapture('test_videos/test_video_home.mp4')

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_videos/output_home.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# List to store embeddings
person_embeddings = []

# Initialize last verification time to current time
last_verification_time = time.time()

# Initialize verification flag and target person's center coordinates
verified = False
center_x, center_y = 0, 0

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
                x1_n, y1_n, x2_n, y2_n = boxes.xyxyn[i]

                # Calculate the absolute center of the bounding box
                center_x = int((x1_n + x2_n) / 2 * frame.shape[1])
                center_y = int((y1_n + y2_n) / 2 * frame.shape[0])

                # Crop the person from the frame
                person0 = frame[int(y1_n*frame.shape[0]):int(y2_n*frame.shape[0]), int(x1_n*frame.shape[1]):int(x2_n*frame.shape[1])]

                # If there are less than 10 images, save the current image and its embedding in list
                if len(person_embeddings) < 30:
                    try:
                        embedding = DeepFace.represent(person0, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                        person_embeddings.append(embedding)
                    except ValueError as e:
                        print("Could not create embedding: ", str(e))

                # If 10 images are collected and at least 5 seconds have passed since the last verification, perform face recognition
                if len(person_embeddings) >= 10 and time.time() - last_verification_time >= 5:
                    print("10 images collected, performing person recognition...")
                    embedding = DeepFace.represent(person0, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                    distances = [np.linalg.norm(np.array(embedding[0]['embedding']) - np.array(person_embedding[0]['embedding'])) for person_embedding in person_embeddings]
                    print("Distances: ", distances)
                    if any(distance < 0.3 for distance in distances):  # Threshold of 0.3, you might need to adjust it
                        print(f"Person detected at ({center_x}, {center_y})")
                        verified = True
                        match_found = True
                        last_verification_time = time.time()
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
