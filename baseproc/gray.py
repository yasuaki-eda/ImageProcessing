import cv2
import numpy as np


def BGR2GRAY(img):
  if len(img.shape) == 3:
    return (img[:,:,0] * 0.0722 + img[:,:,1]*0.7152 + img[:,:,2]*0.2126).astype(np.uint8)
  else:
    print('image channel must be 3ch.')
    return img

if __name__ == '__main__':
  img = cv2.imread('../image/imori.jpg')
  img_g = BGR2GRAY(img)
  cv2.imshow('img_g', img_g)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
