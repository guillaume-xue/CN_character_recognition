import os
import joblib
import time
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
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
        SVC()
    )
    
    model.fit(X_train, y_train)
    print("SVM model trained successfully.")
    return model

def find_best_svm_params(X_train, y_train, cv=5, n_jobs=-1, verbose=2):
    pipeline = make_pipeline(
        StandardScaler(),
        SVC()
    )

    param_grid = {
        'svc__C': [0.1, 1.0, 10.0, 100.0],
        'svc__kernel': ['rbf', 'linear', 'poly'],
        'svc__gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    actual_time = end_time - start_time

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Grid search completed in {timedelta(seconds=int(actual_time))}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")
    return grid_search.cv_results_

def print_results(cv_results):
    means = cv_results['mean_test_score']
    stds = cv_results['std_test_score']
    params = cv_results['params']
    
    for mean, std, param in zip(means, stds, params):
        print(f"Mean accuracy: {mean:.4f} (Std: {std:.4f}) with parameters: {param}")

def train_reglog(X_train, y_train):
    print("Training Logistic Regression model...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")
    return model

def train_randforest(X_train, y_train):
    print("Training Random Forest model...")
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier()
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
    
def load_all_models(X_train, y_train, svm_model=True, logreg_model=True, 
    randforest_model=True):
    os.makedirs(TRAIN_MODEL_DIR, exist_ok=True)

    if svm_model is True:
        svm_model = load_model(SVM_MODEL_FILE)
        if svm_model is None:
            svm_model = train_svm(X_train, y_train)
            save_model(svm_model, SVM_MODEL_FILE)
    
    if logreg_model is True:
        logreg_model = load_model(LOGREG_MODEL_FILE)
        if logreg_model is None:
            logreg_model = train_reglog(X_train, y_train)
            save_model(logreg_model, LOGREG_MODEL_FILE)

    if randforest_model is True:
        randforest_model = load_model(RANDFOREST_MODEL_FILE)
        if randforest_model is None:
            randforest_model = train_randforest(X_train, y_train)
            save_model(randforest_model, RANDFOREST_MODEL_FILE)
    
    return svm_model, logreg_model, randforest_model

def evaluate_one_model(model, X_test, y_test, model_name="Model"):
    print(f"Calculating {model_name} accuracy...")
    predictions = []
    for i in tqdm(range(len(X_test)), desc=f"{model_name} Predictions"):
        predictions.append(model.predict([X_test[i]])[0])
    predictions = np.array(predictions)
    accuracy = sum(predictions == y_test) / len(y_test)
    return accuracy

def evaluate_model(svm_model, logreg_model, randforest_model, X_test, y_test, evaluate_svm=True, evaluate_logreg=True, evaluate_randforest=True):
    accuracies = {}
    
    if evaluate_svm is True:
        svm_accuracy = evaluate_one_model(svm_model, X_test, y_test, model_name="SVM")
    if evaluate_logreg is True:
        logreg_accuracy = evaluate_one_model(logreg_model, X_test, y_test, model_name="Logistic Regression")
    if evaluate_randforest is True:
        randforest_accuracy = evaluate_one_model(randforest_model, X_test, y_test, model_name="Random Forest")

    if evaluate_svm is True:
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
    if evaluate_logreg is True:
        print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")
    if evaluate_randforest is True:
        print(f"Random Forest Accuracy: {randforest_accuracy:.4f}")
    
    return accuracies

def evaluate_one_model_with_cv(model, X_train, y_train, X_test, y_test, model_name="Model", cv=5):
    print(f"\nÉvaluation {model_name} avec {cv}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                scoring='accuracy', n_jobs=-1)
    print(f"{model_name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

def evaluate_model_with_cv(svm_model, logreg_model, randforest_model, X_train, y_train, X_test, y_test, 
                           evaluate_svm=True, evaluate_logreg=True, evaluate_randforest=True, cv=5):
    print("Start cross evaluation ")
    
    if evaluate_svm and svm_model is not None:
        evaluate_one_model_with_cv(svm_model, X_train, y_train, X_test, y_test, model_name="SVM", cv=cv)
    
    if evaluate_logreg and logreg_model is not None:
        evaluate_one_model_with_cv(logreg_model, X_train, y_train, X_test, y_test, model_name="Logistic Regression", cv=cv)
    
    if evaluate_randforest and randforest_model is not None:
        evaluate_one_model_with_cv(randforest_model, X_train, y_train, X_test, y_test, model_name="Random Forest", cv=cv)

    print("End of cross validation evaluation.")