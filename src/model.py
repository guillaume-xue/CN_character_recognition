import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

def train_svm(X_train, y_train):
    print("Training SVM model...")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10.0, verbose=2)
    )
    
    model.fit(X_train, y_train)
    print("SVM model trained successfully.")
    return model

def train_reglog(X_train, y_train):
    print("Training Logistic Regression model...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")
    return model

def train_randforest(X_train, y_train):
    print("Training Random Forest model...")
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)
    )
    
    model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")
    return model

def save_model(model, filename):
    joblib.dump(model, filename)
    print("Model saved to" + filename)

def load_model(filename):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print("Model loaded from" + filename)
        return model
    else:
        print("Model file " + filename + " not found")
        return None