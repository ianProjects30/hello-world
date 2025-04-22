from ultralytics import YOLO
import cv2

use_grayscale = True
use_bilateral_filter = True
use_background_subtraction = False 

model = YOLO('C:/Users/Ian/Downloads/My Thesis/runs/detect/dectionv23/weights/best.pt')

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if use_grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if use_bilateral_filter:
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    if use_background_subtraction:
        fgmask = fgbg.apply(frame)
        frame = cv2.bitwise_and(frame, frame, mask=fgmask)

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("cnn", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

