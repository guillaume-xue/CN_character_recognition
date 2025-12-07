import src.data_loader as data_loader
import src.model as model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.loadImages()
    
    # train or load models
    svm_model, logreg_model, randforest_model = model.load_all_models(X_train, y_train)

    # test the models
    model.evaluate_model(svm_model, logreg_model, randforest_model, X_test, y_test)