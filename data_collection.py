import cv2
import mediapipe as mp
import csv

class DataCollector:
    def __init__(self, filename='landmark_data.csv'):
        self.filename = filename
        self.header_written = False

    def save_sample(self, landmarks, label):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not self.header_written:
                header = [f"feature_{i}" for i in range(len(landmarks))] + ["label"]
                writer.writerow(header)
                self.header_written = True
            writer.writerow(landmarks + [label])

def main():
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    
    collector = DataCollector('landmark_data.csv')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible!")

    print("Press 's' to save a sample and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
               
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                cv2.putText(frame, "Press 's' to save sample", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and results.multi_hand_landmarks:
          
            label = input("Enter label for this sample (e.g., 'A'): ")
           
            collector.save_sample(landmarks, label)
            print(f"Sample saved for label '{label}'.")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
