import cv2
import numpy as np

from ultralytics import YOLO
model = YOLO("best.pt")

def bbox(boxes, class_name):
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w = x2 - x1
        d_payload = (15*320)/w
        d_bucket = (30*320)/w
        if(class_name=="payload"):
            cv2.putText(frame, class_name, (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Depth: {int(d_payload)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
        if(class_name=="bucket"):
            cv2.putText(frame, class_name, (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Depth: {int(d_bucket)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
# webcam ver
cap = cv2.VideoCapture(0)
fps = cv2.CAP_PROP_FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) 
while True:
    ret, frame = cap.read()
    results_ember = model.predict(source=frame, save=False, save_txt=False, classes = [0], conf=0.7, verbose=False)
    results_payload = model.predict(source=frame, save=False, save_txt=False, classes = [1], conf=0.7, verbose=False)

    for r in results_ember:
        boxes = r.boxes
        bbox(boxes, "bucket")

    for r in results_payload:
        boxes = r.boxes
        bbox(boxes, "payload")
    
    cv2.putText(frame, str(fps) + " FPS", (0,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,0,0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27 : break

cap.release()
cv2.destroyAllWindows()