import cv2
from ultralytics import YOLO

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

                print(f"Center of normalized bounding box {i}: ({center_x}, {center_y})")

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
