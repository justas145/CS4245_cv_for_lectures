import cv2
import os
import time
from ultralytics import YOLO
from deepface import DeepFace

# Load face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model
model = YOLO('yolov8n.pt')

# Start capturing the video file
cap = cv2.VideoCapture('test_video_lecturer_and_people.mp4')

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_lecturer_and_people.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# List to store face images
face_images = []

# Initialize last verification time to current time
last_verification_time = time.time()
verified = False

# CameraPTZ class in order to keep track of the camera's position and facilitate movement
class CameraPTZ:
    def __init__(self, camera):
        self.camera = camera
        self.pan = 0
        self.tilt = 0
        self.zoom = 0

    def move(self, pan, tilt, zoom):
        self.pan += pan
        self.tilt += tilt
        self.zoom += zoom

        # Limit pan to [-180, 180]
        if self.pan > 180:
            self.pan = 180
        elif self.pan < -180:
            self.pan = -180

        # Limit tilt to [-90, 90]
        if self.tilt > 90:
            self.tilt = 90
        elif self.tilt < -90:
            self.tilt = -90

        # Limit zoom to [0, 100]
        if self.zoom > 100:
            self.zoom = 100
        elif self.zoom < 0:
            self.zoom = 0


while True:  # You want continuous detection
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the model to the frame
    results = model(frame)

    # Create a CameraPTZ object
    camera = CameraPTZ(0)

    # Process the results
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            if boxes.cls[i].item() == 0:  # 'person'
                # Get normalized bounding box coordinates
                x1_n, y1_n, x2_n, y2_n = boxes.xyxyn[i]

                # Calculate the absolute center of the bounding box
                center_x = int((x1_n + x2_n) / 2 * frame.shape[1])
                center_y = int((y1_n + y2_n) / 2 * frame.shape[0])

                # Crop the person from the frame
                person0 = frame[int(y1_n*frame.shape[0]):int(y2_n*frame.shape[0]), int(x1_n*frame.shape[1]):int(x2_n*frame.shape[1])]

                # Face detection on the cropped person
                faces = face_cascade.detectMultiScale(person0, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # If a face is detected
                for (x, y, w, h) in faces:
                    # Crop the face from the person
                    face0 = person0[y:y+h, x:x+w]

                    # If there are less than 10 images, save the current image in list
                    if len(face_images) < 10:
                        face_images.append(face0)
                        continue

                # If 10 images are collected and at least 5 seconds have passed since the last verification, perform face recognition
                if len(face_images) >= 10 and time.time() - last_verification_time >= 5:
                    print("10 images collected, performing face recognition...")
                    for detected_face in face_images:
                        result = DeepFace.verify(detected_face, face0, model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)

                        # If the result is verified, print the location of the person in the frame
                        if result['verified']:
                            print(f"Person detected at ({center_x}, {center_y})")
                            verified = True
                            last_verification_time = time.time()

                            # If the person is close to 5% near the edge of the frame, move the camera (left and right only for now)
                            if center_x < 0.05 * frame.shape[1]:
                                print("Move camera left")
                                camera.move(-10, 0, 0)
                            elif center_x > 0.95 * frame.shape[1]:
                                print("Move camera right")
                                camera.move(10, 0, 0)
                            break
                        else:
                            print("Person not detected")
                            verified = False

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Draw a green circle if person is verified, red otherwise
    if verified:
        cv2.circle(annotated_frame, (50,50), 25, (0,255,0), -1)
        
        # add a blue circle at the center of verified person
        cv2.circle(annotated_frame, (center_x, center_y), 25, (255,0,0), -1)
        
        # # Add the normalized coordinates at the top right
        # cv2.putText(annotated_frame, f"Normalized Coords: {x1_n:.2f},{y1_n:.2f},{x2_n:.2f},{y2_n:.2f}", (annotated_frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    else:
        cv2.circle(annotated_frame, (50,50), 25, (0,0,255), -1)

    # Display the annotated frame
    #cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Write the frame into the file 'output.mp4'
    out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the VideoCapture and VideoWriter object and close display windows
cap.release()
out.release()
cv2.destroyAllWindows()
