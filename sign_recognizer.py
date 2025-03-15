import os
import joblib
import numpy as np
import pandas as pd

class SignRecognizer:
    def __init__(self, model_path='sign_classifier.pkl'):
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print("Error loading model:", e)
                self.model = None
        else:
            print(f"Model file not found at '{model_path}'.")
            self.model = None

    def predict(self, landmarks):
        if self.model is None:
            return "Model not loaded"
        
        data = np.array(landmarks).reshape(1, -1)
        
        feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=feature_names)
        try:
            prediction = self.model.predict(df)
            return prediction[0]
        except Exception as e:
            print("Error during prediction:", e)
            return "Error"
