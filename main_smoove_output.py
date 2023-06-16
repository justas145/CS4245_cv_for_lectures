from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import build_model, extract_features, extract_faces
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import cv2
import argparse
import time
start_time = time.time()


# Parse command line arguments
parser = argparse.ArgumentParser(description='Perform action recognition')
# parser.add_argument('-model', help='yolo model to use', default='yolov8n')
parser.add_argument(
    '-method', help='method to use for feature extraction', default='osnet')
parser.add_argument('-detection_confidence', type=float,
                    help='detection confidence threshold', default=0.75)
parser.add_argument('-num_saved_images', type=int,
                    help='number of saved images for the target person', default=30)
parser.add_argument('-verification_time', type=float,
                    help='time interval between verifications', default=3)
# parser.add_argument('-similarity_method',
#                     help='method to use for calculating similarity scores', default='cosine')
parser.add_argument('-threshold_coefficient', type=float,
                    help='coefficient for similarity threshold', default=0.9)
parser.add_argument(
    '-save_frames', type = bool, help='save frames with the title of similarity score and yes/no for verification ', default=False)
parser.add_argument('-input_video', help='input video file', type=str)


args = parser.parse_args()
# Get the name of the test video
test_video = args.input_video.split('/')[-1].split('.')[0]

if args.save_frames:
    print('saving frames to saved_frames')
    directory = f'saved_frames/{test_video}'

    # Check if the directory exists, if not, create it.
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load model
model = YOLO('yolov8n.pt')

# Initialize the feature extraction method
if args.method == 'osnet':
    osnet_model = build_model()
    face_cascade = None
elif args.method == 'face_recognition':
    osnet_model = None
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Start capturing the video file
cap = cv2.VideoCapture(f'{args.input_video}')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f'output_videos/{test_video}_{args.method}.mp4',
                      fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# List to store embeddings
embeddings_lst = []

# Initialize last verification time to current time
last_verification_time = time.time()
# Initialize verification flag and target person's center coordinates
verified = False
center_x, center_y = 0, 0
# new variables to store the coordinates of verified person
verified_x, verified_y = 0, 0
min_distance = 50
threshold = 0.5
d = 0
average_similarity = 0
while True:  # You want continuous detection
    # Initialize match flag
    match_found = False
    verification_done = False
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the model to the frame
    results = model(frame)

    if len(embeddings_lst) < args.num_saved_images or time.time() - last_verification_time >= args.verification_time:
        # Process the results
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # single box
                box = boxes[i]
                # 'person' with confidence >= 0.75
                if box.cls[0].item() == 0 and round(box.conf[0].item(), 2) >= args.detection_confidence:
                    # Get normalized bounding box coordinates
                    x1_n, y1_n, x2_n, y2_n = box.xyxyn[0]
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Calculate the absolute center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    # get the frame of the person
                    person0 = frame[int(y1):int(y2), int(x1):int(x2)]
                    person = Image.fromarray(person0)
                    # make embeddings of the person
                    if len(embeddings_lst) < args.num_saved_images:
                        if args.method == 'osnet':
                            # Crop the person from the frame
                            embedding = extract_features(osnet_model, person)
                            embeddings_lst.append(embedding)

                        elif args.method == 'face_recognition':
                            embedding = extract_faces(
                                person0, face_cascade, enforce_detection=True)
                            if embedding is not None:
                                embeddings_lst.append(embedding)

                    # if embeddings are ready, start verification
                    if len(embeddings_lst) >= args.num_saved_images and time.time() - last_verification_time >= args.verification_time:
                        print(
                            f"{args.num_saved_images} images collected, performing identification...")
                        verification_done = True
                        # Extract features from the person and compare them with the target embeddings
                        embedding = extract_features(osnet_model, person) if args.method == 'osnet' else extract_faces(
                            person0, face_cascade, enforce_detection=False)
                        if embedding is None:
                            print('Embedding is None')
                            continue
                        
                        similarities = [cosine_similarity(embedding.reshape(
                            1, -1), embedding_target.reshape(1, -1)) for embedding_target in embeddings_lst]
                        average_similarity = sum(similarities) / len(similarities)
                        average_similarity = average_similarity.item()
                        print(f'Average similarity:, {average_similarity:.2f}')
                        if average_similarity >= threshold:
                            print(
                                f"### PERSON DETECTED ### at ({center_x}, {center_y})")
                            verified = True
                            match_found = True
                            last_verification_time = time.time()
                            # update the coordinates of verified person
                            verified_x, verified_y = center_x, center_y
                            # update the threshold
                            if d == 0:
                                threshold = args.threshold_coefficient * average_similarity
                                print('updating threshold to:', threshold)
                            if args.save_frames:
                                print('saving frames to saved_frames' )
                                cv2.imwrite(
                                    f'saved_frames/{test_video}/person_{d}_{average_similarity:.2f}_YES.jpg', person0)
                            d += 1
                            break
                        else:
                            print("Person not detected")
                            verified = False
                            match_found = False
                            if args.save_frames:
                                print('saving frames to saved_frames' )
                                cv2.imwrite(
                                    f'saved_frames/{test_video}/person_{d}_{average_similarity:.2f}_NO.jpg', person0)
                            d += 1
                        
                    if match_found:
                        break
        if verification_done and not match_found:
            last_verification_time = time.time()
    
    # Visualize the results on the frame
    
    annotated_frame = results[0].plot()

    # Draw a green circle if person is verified, red otherwise
    if verified:
        cv2.circle(annotated_frame, (50, 50), 25, (0, 255, 0), -1)
        # write average similarity on the frame
        cv2.putText(annotated_frame, f'{average_similarity:.2f}',
                    (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # add a blue circle at the center of verified person
        cv2.circle(annotated_frame, (verified_x, verified_y),
                   25, (255, 0, 0), -1)
    else:
        cv2.circle(annotated_frame, (50, 50), 25, (0, 0, 255), -1)

    # Write the frame into the file 'output.mp4'
    out.write(annotated_frame)

# Release the VideoCapture and VideoWriter object and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()
FPS = total_frames / (end_time - start_time)
print(f'Time taken to process the video: {end_time - start_time} seconds.')
print(f'FPS: {FPS}.')

# arguments:
# -model
# -method
# -detection_confidence
# -num_saved_images
# -verification_time
# -similarity_method
# -threshold_coefficient
# -save_frames
