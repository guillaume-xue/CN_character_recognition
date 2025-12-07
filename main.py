import src.data_loader as data_loader
import src.model as model
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.loadImages()
    
    svm_model = model.train_svm(X_train, y_train)
    logreg_model = model.train_reglog(X_train, y_train)

    # test the models
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
    
    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"Logistic Regression Accuracy: {logreg_accuracy}")