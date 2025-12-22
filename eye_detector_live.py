# git clone
Step 2.5: Cascade classifier for eye detection

from picamera2 import Picamera2
import cv2, time

# Load Haar eye cascade from OpenCV's built-in data path
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)     # use (1280, 720) if you want higher res
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(0.3)  # warm-up

try:
    while True:
        # Capture frame (RGB) and convert to BGR for OpenCV
        frame = picam2.capture_array()

        # Grayscale for Haar detection (equalize helps in variable lighting)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect eyes; tune these params for your scene
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(20, 20)
        )

        # Draw eye boxes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # (Optional) overlay count
        cv2.putText(frame, f"Eyes: {len(eyes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("", frame)

        # Quit on 'q' or ESC
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
