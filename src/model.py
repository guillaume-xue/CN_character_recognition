import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np

TRAIN_MODEL_DIR = 'data/train_model/'
SVM_MODEL_FILE = TRAIN_MODEL_DIR + 'svm_model.pkl'
LOGREG_MODEL_FILE = TRAIN_MODEL_DIR + 'logreg_model.pkl'
RANDFOREST_MODEL_FILE = TRAIN_MODEL_DIR + 'randforest_model.pkl'

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
    
def load_all_models(X_train, y_train):
    svm_model = load_model(SVM_MODEL_FILE)
    if svm_model is None:
        svm_model = train_svm(X_train, y_train)
        save_model(svm_model, SVM_MODEL_FILE)
    
    logreg_model = load_model(LOGREG_MODEL_FILE)
    if logreg_model is None:
        logreg_model = train_reglog(X_train, y_train)
        save_model(logreg_model, LOGREG_MODEL_FILE)

    randforest_model = load_model(RANDFOREST_MODEL_FILE)
    if randforest_model is None:
        randforest_model = train_randforest(X_train, y_train)
        save_model(randforest_model, RANDFOREST_MODEL_FILE)

    return svm_model, logreg_model, randforest_model

def evaluate_model(svm_model, logreg_model, randforest_model, X_test, y_test):
    print("Calculating SVM accuracy...")
    predictions = []
    for i in tqdm(range(len(X_test)), desc="SVM Predictions"):
        predictions.append(svm_model.predict([X_test[i]])[0])
    predictions = np.array(predictions)
    svm_accuracy = sum(predictions == y_test) / len(y_test)
    
    print("Calculating Logistic Regression accuracy...")
    predictions = []
    for i in tqdm(range(len(X_test)), desc="Logistic Regression Predictions"):
        predictions.append(logreg_model.predict([X_test[i]])[0])
    predictions = np.array(predictions)
    logreg_accuracy = sum(predictions == y_test) / len(y_test)

    print("Calculating Random Forest accuracy...")
    predictions = []
    for i in tqdm(range(len(X_test)), desc="Random Forest Predictions"):
        predictions.append(randforest_model.predict([X_test[i]])[0])
    predictions = np.array(predictions)
    randforest_accuracy = sum(predictions == y_test) / len(y_test)

    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"Logistic Regression Accuracy: {logreg_accuracy}")
    print(f"Random Forest Accuracy: {randforest_accuracy}")