import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def main():
    data = pd.read_csv('landmark_data.csv')
    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    
    joblib.dump(clf, 'sign_classifier.pkl')
    print("Classifier saved as 'sign_classifier.pkl'.")

if __name__ == "__main__":
    main()