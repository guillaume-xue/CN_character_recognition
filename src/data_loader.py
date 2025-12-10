import pickle
import csv
import random
import cv2
from pathlib import Path
import numpy as np
import os
import src.features as features

DATA_DIR_1 = "data/data"
CSV_FILE_1 = "data/chinese_mnist.csv"

DATA_DIR_2 = "data/chinese-handwriting/CASIA-HWDB_"
CSV_FILE_2 = "data/chinese_handwriting.csv"

DATA_LOAD_PATH = "data/load_data/"
HOG_IMAGES_FILE = DATA_LOAD_PATH + "hog_images.npy"
HOG_LABELS_FILE = DATA_LOAD_PATH + "hog_labels.npy"
HOG_DATA_FILE = DATA_LOAD_PATH + "hog_data.pkl"

def read_csv(dataset_type):
  data = []
  csv_file = CSV_FILE_1 if dataset_type == '1' else CSV_FILE_2
  with open(csv_file, 'r') as file:
    read = csv.reader(file)
    for line in read:
      data.append(line)
  return data

def get_image_path(index, dataset_type):
  if dataset_type == '1':
    img_path = f"{DATA_DIR_1}/input_{index[0]}_{index[1]}_{index[2]}.jpg"
  else:
    img_path = f"{DATA_DIR_2}{index[4]}/{index[4]}/{index[0]}/{index[1]}.png"
  return img_path

def get_label_index(header):
  for idx, col_name in enumerate(header):
    if col_name == "code":
      return idx
  return -1

def load_data_with_hog(dataset):
  read = read_csv(dataset)
  label_idx = get_label_index(read[0])
  read = read[1:]
  
  images = []
  labels = []

  for line in read:
    img_path = get_image_path(line, dataset)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      continue
    image = cv2.resize(image, (64, 64))
    hog_feat = features.extract_hog_features(image)
    images.append(hog_feat)
    labels.append(int(line[label_idx]))

  print(f"Loaded {len(images)} images.")

  os.makedirs(DATA_LOAD_PATH, exist_ok=True)

  np.save(HOG_IMAGES_FILE, np.array(images))
  np.save(HOG_LABELS_FILE, np.array(labels))

  with open(HOG_DATA_FILE, 'wb') as f:
    pickle.dump((images, labels), f)

  return images, labels

def split_data_randomly(images, labels, seed=1234):
  print("Splitting data into train and test sets...")

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

# -----------------------------------------------------------------

def create_csv_file():
  with open(CSV_FILE_2, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["dir","index","code", "character", "dataset_type"])

    code = 0

    for dataset_type in ['Test', 'Train']:
      base_path = Path(f"{DATA_DIR_2}{dataset_type}/{dataset_type}/")
      
      for char_dir in sorted(base_path.iterdir()):
        dir_name = char_dir.name
        png_files = sorted(char_dir.glob('*.png'))

        for img_file in png_files:
          index = img_file.stem
          writer.writerow([dir_name, index, code, dir_name, dataset_type])

        code += 1
      code = 0
      
  print(f"CSV file created at {CSV_FILE_2}")

#------------------------------------------------------------------

def load_images(dataset='1'):
  try:
    with open(HOG_DATA_FILE, 'rb') as f:
      images, labels = pickle.load(f)
    print("Loaded data from pickle file.")
  except (FileNotFoundError, EOFError):
    print("Pickle file not found or empty, loading data afresh.")
    images, labels = load_data_with_hog(dataset)

  return split_data_randomly(images, labels)