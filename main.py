import cv2
from ultralytics import YOLO
from deepface import DeepFace

# load model
model = YOLO('yolov8n.pt')

# Start capturing the webcam feed
cap = cv2.VideoCapture(0)

while True:  # You want continuous detection
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
            if boxes.cls[i].item() == 0:  # 'person'
                # Get normalized bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxyn[i]

                # Calculate the normalized center of the bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Crop the face from the frame
                face = frame[int(y1):int(y2), int(x1):int(x2)]

                # Perform face recognition using DeepFace
                # We pass the frame to DeepFace when Yolo detects a person in the frame
                # DeepFace verify function then compares the face in the frame, to a face in a known saved image,
                # and returns a result of whether the face in the frame is the same as the face in the known image

                # DeepFace uses the VGG-Face model at this moment

                # detected_face = DeepFace.extract_faces(img_path=face, detector_backend='opencv', enforce_detection=False)
                # (face, detector_backend='opencv', enforce_detection=False)
                
                # Verify if person in frame is Bryan
                result = DeepFace.verify(frame, 'deepface_data/images/bryan/bryan1.jpg', model_name='VGG-Face', detector_backend='opencv', enforce_detection=False)
                
                # If the result is verified, print the location of the person in the frame
                if result['verified']:
                    print(f"Bryan detected at ({center_x}, {center_y})")

    # Display the frame
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close display windows
cap.release()
cv2.destroyAllWindows()
