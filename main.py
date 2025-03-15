import cv2
from hand_detector import HandDetector
from sign_recognizer import SignRecognizer

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible")
        return

    detector = HandDetector(max_num_hands=1, detection_confidence=0.7)
    recognizer = SignRecognizer(model_path='sign_classifier.pkl')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        if results.multi_hand_landmarks:
          
            frame = detector.draw_hands(frame, results)
        
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = detector.extract_landmarks(hand_landmarks)
                if len(landmarks) > 0:
                    sign = recognizer.predict(landmarks)
                    cv2.putText(frame, f'Sign: {sign}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                break 

        cv2.imshow("Sign Language Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
