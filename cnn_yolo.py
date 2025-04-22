import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

# === CNN Classifier ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x

# Load CNN model
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()

# CNN transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Class mapping
cnn_classes = ['face', 'plastic_bottle']

# Load YOLOv8 model
yolo_model = YOLO('C:/Users/Ian/Downloads/My Thesis/runs/detect/dectionv23/weights/best.pt')

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        try:
            # Prepare cropped image for CNN
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0)

            # Classify with CNN
            with torch.no_grad():
                outputs = cnn_model(input_tensor)
                _, pred = torch.max(outputs, 1)
                label = cnn_classes[pred.item()]
        except Exception as e:
            label = "error"

        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + CNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
