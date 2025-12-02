from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_svm(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0)
    )
    
    model.fit(X_train, y_train)

    return model