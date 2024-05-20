import cv2
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Load YOLO network
net = cv2.dnn.readNet("dataset/custom-yolov4-tiny-detector_final.weights", "dataset/custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Open webcam capture (use 0 for the default webcam)
vc = cv2.VideoCapture(2)

if not vc.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        print("Error: Frame not grabbed.")
        break

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # Initialize drawing time variables
    start_drawing = time.time()
    end_drawing = start_drawing

    # Draw bounding boxes if any objects are detected
    if len(classes) > 0:
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            if(score > 0.99):
                color = COLORS[int(classid) % len(COLORS)]
                label = "ClassID %d : %f" % (classid, score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

    # Display FPS
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show the frame
    cv2.imshow("detections", frame)

vc.release()
cv2.destroyAllWindows()
