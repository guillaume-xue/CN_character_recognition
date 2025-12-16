import cv2
import matplotlib.pyplot as plt
import src.data_loader as data_loader
import src.model as model
import src.features as features

TRAIN_MODEL_DIR = 'data/train_model/'
SVM_MODEL_FILE = TRAIN_MODEL_DIR + 'svm_model.pkl'
LOGREG_MODEL_FILE = TRAIN_MODEL_DIR + 'logreg_model.pkl'
RANDFOREST_MODEL_FILE = TRAIN_MODEL_DIR + 'randforest_model.pkl'

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = data_loader.load_images()

    # Train the model
    best_svm, best_params, results = model.find_best_svm_params(X_train, y_train, cv=5)

    # Use the best model for predictions
    predictions = best_svm.predict(X_test)
    # features.display_features(type=1)