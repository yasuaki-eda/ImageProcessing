import cv2
import numpy as np
import sys
sys.path.append('../baseproc')
import gray


def make_noise(img, seed=0, rate=0.01):
  if seed is not None:
    np.random.seed(seed)
  if len(img.shape) == 3:
    h, w, c = img.shape
    x = np.tile(range(h), w).reshape(w, h).T
    y = np.tile(range(w), h).reshape(h, w)
    rnum = int(h * w * rate)
    choice = np.random.choice(range(h*w), rnum, replace=False)
    for z in choice:
      xx = x[z//w, z%w]
      yy = y[z//w, z%w]
      rch = np.random.randint(3)
      val = np.random.randint(255)
      img[xx, yy, rch] = val
    return img
  else :
    h, w = img.shape
    x = np.tile(range(h), w).reshape(w, h).T
    y = np.tile(range(w), h).reshape(h, w)
    rnum = int(h * w * rate)
    choice = np.random.choice(range(h*w), rnum, replace=False)
    for z in choice:
      xx = x[z//w, z%w]
      yy = y[z//w, z%w]
      val = np.random.randint(255)
      img[xx, yy] = val
    return img

def test_make_noise(img):
  img_g = gray.BGR2GRAY(img)
  dst = make_noise(img_g, seed=None, rate=0.01)
  if dst is not None:
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
  img = cv2.imread('../image/thorino.jpg')
  test_make_noise(img)
