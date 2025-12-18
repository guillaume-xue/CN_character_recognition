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

    svm, rand, reg = model.load_all_models(X_train, y_train)

    acc = model.evaluate_model(svm, reg, rand, X_test, y_test)