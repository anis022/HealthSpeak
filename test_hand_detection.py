import cv2
from hand_detector import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        frame = detector.draw_hands(frame, results)

        cv2.imshow("Hand Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
