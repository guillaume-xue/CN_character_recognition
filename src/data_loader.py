import csv
import cv2
import features

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
    if image is not None:
      image = cv2.resize(image, (64, 64)) 
      hog_feat = features.extract_hog_features(image)
    images.append(hog_feat)
    labels.append(line[3])
  print(f"Loaded {len(images)} images with labels.")
  return images, labels