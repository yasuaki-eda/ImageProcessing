import cv2
import numpy as np
from gray import BGR2GRAY
import math

def quanta(img, qnum=4):
  qstep = math.ceil(255/qnum)
  for q in range(qnum):
    print(q, q*qstep, (q+1)*qstep)
    img = np.where( (q*qstep<=img) & (img<(q+1)*qstep), q*qstep, img)
  return img.astype(np.uint8)


def binarization(img, threshold=128):
  img_g = BGR2GRAY(img)
  return np.where(img_g<threshold, 0, 255).astype(np.uint8)

def test_bin(img):
  dst = binarization(img, threshold=200)
  if dst is not None:
    cv2.imshow('bin', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_quanta(img):
  dst = quanta(img, qnum=5)
  if dst is not None:
    cv2.imshow('quanta', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
  img = cv2.imread('../image/imori.jpg')
#  test_bin(img)  
  test_quanta(img)

