import torch
import cv2
import time

model_payload = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5.pt')
model_bucket = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo5.pt')
model_payload.classes = 1
model_bucket.classes = 0

def bbox(boxes, class_name):
    for box in boxes:
        x1, y1, x2, y2, _, _ = map(int, box)
        w = x2 - x1
        d_payload = (15*320)/w
        d_bucket = (30*320)/w
        if(class_name=="payload"):
            cv2.putText(frame, class_name, (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Depth: {int(d_payload)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        if(class_name=="bucket"):
            cv2.putText(frame, class_name, (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Depth: {int(d_bucket)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) 
prev_time = time.time()

while True:
    ret, frame = cap.read()

    results_payload = model_payload(frame)
    results_bucket = model_bucket(frame)

    bbox(results_bucket.xyxy[0], "bucket")
    bbox(results_payload.xyxy[0], "payload")
    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(frame, str(int(fps)) + " FPS", (0,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()