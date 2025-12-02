import cv2
from skimage.feature import hog

def crop_and_resize(image, target_size=(64, 64)):
  coords = cv2.findNonZero(image) 
  if coords is None:
    return image 
  x, y, w, h = cv2.boundingRect(coords)
  cropped = image[y:y+h, x:x+w]
  resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
  return resized

def extract_hog_features(image):
  image = crop_and_resize(image)
  features = hog(image, 
                  orientations=9, 
                  pixels_per_cell=(4, 4), 
                  cells_per_block=(2, 2), 
                  block_norm='L2-Hys', 
                  visualize=False)
  return features