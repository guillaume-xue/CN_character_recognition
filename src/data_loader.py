import pickle
import csv
import random
import cv2
import numpy as np
import src.features as features

DATA_DIR = "data/data"
CSV_FILE = "data/chinese_mnist.csv"

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

  np.save('data/preload/hog_images.npy', np.array(images))
  np.save('data/preload/hog_labels.npy', np.array(labels))

  with open('data/preload/hog_data.pkl', 'wb') as f:
    pickle.dump((images, labels), f)

  return images, labels

def split_data_randomly(images, labels, seed=1234):
  random.seed(seed)

  combined = list(zip(images, labels))
  random.shuffle(combined)
  images[:], labels[:] = zip(*combined)

  train_images = images[:10000]
  train_labels = labels[:10000]
  test_images = images[10000:15000]
  test_labels = labels[10000:15000]

  return train_images, train_labels, test_images, test_labels

def loadImages():
  try:
    with open('data/preload/hog_data.pkl', 'rb') as f:
      images, labels = pickle.load(f)
    print("Loaded data from pickle file.")
  except (FileNotFoundError, EOFError):
    print("Pickle file not found or empty, loading data afresh.")
    images, labels = load_data_with_hog()

  return split_data_randomly(images, labels)