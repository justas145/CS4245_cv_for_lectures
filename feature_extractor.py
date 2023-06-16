import torchreid
from PIL import Image
import torch
from torchvision import transforms
from deepface import DeepFace
import numpy as np
def build_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    model.eval()
    return model


def extract_features(model, image):
    # Convert your image to a tensor and normalize it
    transform = transforms.Compose([
        # OSNet typically uses an input size of 256x128
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        # values for ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = transform(image)  # Assuming the image is a PIL Image
    image = image.unsqueeze(0)  # Add a batch dimension

    # Extract features
    with torch.no_grad():
        features = model(image)

    return features.numpy()


def extract_faces(image, face_cascade, enforce_detection):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        # will always return embedding if enforce_detection=False
        try:
            embedding = DeepFace.represent(face, model_name='VGG-Face', detector_backend='opencv', enforce_detection=enforce_detection)
            print("Face detected, collecting images...")
            # turn to np array
            embedding = np.array(embedding[0]['embedding'])
            return embedding
        except ValueError:
            print("ValueError encountered, skipping face...")
            continue
    return None

