import pickle
import csv
import random
import cv2
import numpy as np
import src.features as features

DATA_DIR = "data/data"
CSV_FILE = "data/chinese_mnist.csv"
DATA_LOAD_PATH = "data/load_data/"
HOG_IMAGES_FILE = DATA_LOAD_PATH + "hog_images.npy"
HOG_LABELS_FILE = DATA_LOAD_PATH + "hog_labels.npy"
HOG_DATA_FILE = DATA_LOAD_PATH + "hog_data.pkl"

def readCSV():
  data = []
  with open(CSV_FILE, 'r') as file:
    read = csv.reader(file)
    for line in read:
      data.append(line)
  return data

def load_data_with_hog():
  read = readCSV()[1:]
  
  images = []
  labels = []

  for line in read:
    img_path = f"{DATA_DIR}/input_{line[0]}_{line[1]}_{line[2]}.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      continue
    image = cv2.resize(image, (64, 64))
    hog_feat = features.extract_hog_features(image)
    images.append(hog_feat)
    labels.append(int(line[3]))

  print(f"Loaded {len(images)} images.")

  np.save(HOG_IMAGES_FILE, np.array(images))
  np.save(HOG_LABELS_FILE, np.array(labels))

  with open(HOG_DATA_FILE, 'wb') as f:
    pickle.dump((images, labels), f)

  return images, labels

def split_data_randomly(images, labels, seed=1234):
  random.seed(seed)

  combined = list(zip(images, labels))
  random.shuffle(combined)
  images[:], labels[:] = zip(*combined)

  split_index = int(0.8 * len(images))

  train_images = images[:split_index]
  train_labels = labels[:split_index]
  test_images = images[split_index:]
  test_labels = labels[split_index:]

  return train_images, train_labels, test_images, test_labels

def loadImages():
  try:
    with open(HOG_DATA_FILE, 'rb') as f:
      images, labels = pickle.load(f)
    print("Loaded data from pickle file.")
  except (FileNotFoundError, EOFError):
    print("Pickle file not found or empty, loading data afresh.")
    images, labels = load_data_with_hog()

  return split_data_randomly(images, labels)