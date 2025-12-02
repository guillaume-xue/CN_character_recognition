import src.data_loader as data_loader
import src.model as model

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = data_loader.loadImages()
    svm_model = model.train_svm(train_images, train_labels)