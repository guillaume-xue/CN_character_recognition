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