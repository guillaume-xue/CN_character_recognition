import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

EXEMPLE_IMAGE_PATH_1 = 'data/data/input_69_4_15.jpg'
EXEMPLE_IMAGE_PATH_2 = 'data/chinese-handwriting/CASIA-HWDB_Train/Train/这/1.png'


def crop_image(image):
    coords = cv2.findNonZero(image) 
    if coords is None:
      return image 
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped

def resize_image(image, target_size=(128, 128)):
  resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
  return resized

def binarize_otsu(image):
  _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return binary
  
def hog_features(image, visualize=False):
  if visualize:
    features, hog_image = hog(image, 
                              orientations=12, 
                              pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), 
                              block_norm='L2-Hys', 
                              visualize=True)
    hog_image = hog_image.astype('uint8')
    return features, hog_image
  else:
    features = hog(image, 
                    orientations=12, 
                    pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), 
                    block_norm='L2-Hys')
    return features, None

def preprocess(image):
    binarized = binarize_otsu(image)
    crop = crop_image(binarized)
    resized = resize_image(crop)
    features, _ = hog_features(resized)
    return features

def display_features(type=1):

  image = cv2.imread(EXEMPLE_IMAGE_PATH_1 if type == 1 else EXEMPLE_IMAGE_PATH_2, cv2.IMREAD_GRAYSCALE)
  binarized = binarize_otsu(image)
  crop = crop_image(binarized)
  resized = resize_image(crop, target_size=(128, 128))
  _, hog_image = hog_features(resized, visualize=True)

  _, axes = plt.subplots(1, 5, figsize=(18, 3))
  
  axes[0].imshow(image, cmap='gray')
  axes[0].set_title('Image originale')
  axes[0].axis('off')

  axes[1].imshow(binarized, cmap='gray')
  axes[1].set_title('Image binarisée (Otsu)')
  axes[1].axis('off')

  axes[2].imshow(crop, cmap='gray')
  axes[2].set_title('Image recadrée')
  axes[2].axis('off')

  axes[3].imshow(resized, cmap='gray')
  axes[3].set_title('Image redimensionnée')
  axes[3].axis('off')

  axes[4].imshow(hog_image, cmap='gray')
  axes[4].set_title('Features HOG')
  axes[4].axis('off')

  plt.tight_layout()
  plt.show()