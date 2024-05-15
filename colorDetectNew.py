import cv2
import numpy as np
import time

def empty(a):
    pass

# --- BUAT BIKIN TRACKBAR 
cv2.namedWindow("trackbar")
cv2.createTrackbar("hue_min", "trackbar", 0, 255, empty)
cv2.createTrackbar("sat_min", "trackbar", 0, 255, empty)
cv2.createTrackbar("val_min", "trackbar", 0, 255, empty)
cv2.createTrackbar("hue_max", "trackbar", 255, 255, empty)
cv2.createTrackbar("sat_max", "trackbar", 255, 255, empty)
cv2.createTrackbar("val_max", "trackbar", 255, 255, empty)

def trackbar():
    hmin = cv2.getTrackbarPos("hue_min", "trackbar")
    smin = cv2.getTrackbarPos("sat_min", "trackbar")
    vmin = cv2.getTrackbarPos("val_min", "trackbar")

    hmax = cv2.getTrackbarPos("hue_max", "trackbar")
    smax = cv2.getTrackbarPos("sat_max", "trackbar")
    vmax = cv2.getTrackbarPos("val_max", "trackbar")

    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    return lower, upper

# --- MASKING
def detection(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def main():
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 1)
            lower, upper = trackbar()
            frame = detection(frame, lower, upper)
            
            # --- CALCULATE FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q") or key == ord("Q"):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
