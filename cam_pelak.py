# در این فایل یک نمونه کد برای اتصال به وبکم و شناسایی و برگرداندن پلاک موجود در تصویر وبکم است



from contextlib import suppress
from ultralytics import YOLO
from hezar.models import Model
import cv2

# Load models
lp_detector = YOLO('lp_detector.pt')
lp_detector.verbose = False  # Disable YOLO's default printing
lp_ocr = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


def is_window_open(window_name):
    """Check if a window is open and visible."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 1.0
    except:
        return False

while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Check raw feed for debugging
    cv2.imshow('Raw Webcam Feed', img)

    # Detect plate using YOLOv8 model
    detection = lp_detector(img, conf=0.3, verbose=False)[0]  # Lower confidence, silent mode

    with suppress(Exception):
        if detection.boxes.data.tolist():  # Check if any detections exist
            plate = detection.boxes.data.tolist()[0]

            # Crop plate
            x1, y1, x2, y2, score, class_id = plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            plate_cropped = img[y1:y2, x1:x2]

            # Draw rectangle around plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Use OCR to detect plate characters
            plate_text = lp_ocr.predict(plate_cropped)

            # Display outputs
            print("Detected Plate Text:", plate_text)
            cv2.imshow('Detected Plate', plate_cropped)
        else:
            if is_window_open('Detected Plate'):
                cv2.destroyWindow('Detected Plate')
            # print("No plate detected in this frame.")

    # cv2.imshow('Webcam Feed', img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()