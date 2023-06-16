import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image
from feature_extractor import build_model, extract_features
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = YOLO('yolov8n.pt')

# Initialize OSNet model for feature extraction
osnet_model = build_model()


# Start capturing the video file
cap = cv2.VideoCapture('test_videos/test_video_3.mp4')

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_videos/output_3.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# List to store embeddings
person_embeddings = []

# Initialize last verification time to current time
last_verification_time = time.time()

# Initialize verification flag and target person's center coordinates
verified = False
center_x, center_y = 0, 0
verified_x, verified_y = 0, 0  # new variables to store the coordinates of verified person
min_distance = 50
threshold = 0.5
d = 0
average_similarity = 0
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
                # save the cropped person
                person = Image.fromarray(person0)  # Convert numpy array to PIL Image
                if len(person_embeddings) < 30:
                    print("Collecting images for person recognition...")
                    # Extract features from the cropped person
                    embedding = extract_features(osnet_model, person)
                    person_embeddings.append(embedding)
                
                if len(person_embeddings) >= 30 and time.time() - last_verification_time >= 3:
                    print("10 images collected, performing person recognition...")
                    embedding = extract_features(osnet_model, person)
                    
                    similarities = [cosine_similarity(embedding.reshape(1, -1), person_embedding.reshape(1, -1)) for person_embedding in person_embeddings]
                    average_similarity = sum(similarities) / len(similarities)
                    average_similarity = average_similarity.item()
                    print(f'Average similarity:, {average_similarity}')
                    
                    if average_similarity >= threshold:
                        print(f"### PERSON DETECTED ### at ({center_x}, {center_y})")
                        verified = True
                        match_found = True
                        last_verification_time = time.time()
                        verified_x, verified_y = center_x, center_y  # update the coordinates of verified person
                        if d==0:
                            threshold = 0.9*average_similarity
                            
                        #cv2.imwrite(f'person_frames/person_{d}_{average_similarity:.2f}_YES.jpg', person0)

                        d+=1
                        break

                    else:
                        print("Person not detected")
                        verified = False
                        match_found = False
                        #cv2.imwrite(f'person_frames/person_{d}_{average_similarity:.2f}_NO.jpg', person0)
                        d+=1
                if match_found:
                    break

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Draw a green circle if person is verified, red otherwise
    if verified:
        cv2.circle(annotated_frame, (50,50), 25, (0,255,0), -1)
        # write average similarity on the frame
        cv2.putText(annotated_frame, f'{average_similarity:.2f}', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # add a blue circle at the center of verified person
        cv2.circle(annotated_frame, (verified_x, verified_y), 25, (255,0,0), -1)
    else:
        cv2.circle(annotated_frame, (50,50), 25, (0,0,255), -1)

    # Write the frame into the file 'output.mp4'
    out.write(annotated_frame)

# Release the VideoCapture and VideoWriter object and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()

