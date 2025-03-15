import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible!")
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture frame from webcam!")
    cv2.imshow("Webcam Test", frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
