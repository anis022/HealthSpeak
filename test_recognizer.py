from sign_recognizer import SignRecognizer

sample_landmarks = [0.5, 0.5] * 21 
recognizer = SignRecognizer()

prediction = recognizer.predict(sample_landmarks)
print("Predicted sign:", prediction)
