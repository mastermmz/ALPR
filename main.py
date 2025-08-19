# در این فایل یک نمونه کد برای خواندن عکس های موجود در پوشه دیتابیس و تست میزان دقت تشخیص پلاک نوشته شده است

from contextlib import suppress
from ultralytics import YOLO
from hezar.models import Model
import cv2
import os

lp_detector = YOLO('lp_detector.pt')
lp_ocr = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")


for i in os.listdir('./dataset'):

    # load image
    img = cv2.imread(f'./dataset/{i}')
    
    # detect plate using YoloV8 model
    detection = lp_detector(img)[0]
    
    with suppress(Exception):
        plate = detection.boxes.data.tolist()[0]  

        # crop plate
        x1, y1, x2, y2, score, class_id = plate
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        plate_cropped = img[y1:y2, x1:x2]
        
        # draw rectange around plate
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # using OCR to detect plate characters
        plate_text = lp_ocr.predict(plate_cropped)
        
        # outputs
        print(plate_text)
        cv2.imshow('image', img)
        cv2.imshow('plate', plate_cropped)
        cv2.waitKey()
