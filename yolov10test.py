import cv2 
import time
from ultralytics import YOLO

model = YOLO("dataset/custom_yolov10.pt")

def predict(chosen_model, frame, classes=[]):
    if classes:
        results = chosen_model.predict(frame, classes=classes)
    else:
        results = chosen_model.predict(frame)
    return results

def predict_and_detect(chosen_model, frame, classes=[], rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, frame, classes)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(frame, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return frame, results

# read the image
# image = cv2.imread("testImg/payload_img.jpg")
# result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

# cv2.imshow("Image", result_img)
# cv2.waitKey(0)

#detect video
video_path = r"testVid/payload_vid.mp4"
cap = cv2.VideoCapture(video_path)
while True:
    success, img = cap.read()
    if not success:
        break
    start = time.time()
    result_img, _ = predict_and_detect(model, img, classes=[])
    end = time.time()

    start_drawing = time.time()
    end_drawing = start_drawing

    fps_label = "FPS: %.2f" % (1 / (end - start))
    cv2.putText(result_img, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Video", result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()