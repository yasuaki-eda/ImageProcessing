'''
  エッジ抽出を行います。
  エッジ抽出方法
    sobel, canny, laplacian
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_gradients/py_gradients.html
'''

import cv2
import os
import logging
import numpy as np
import math

OUT_DIR = '..' + os.path.sep + 'out'
INPUT_DIR = '..' + os.path.sep + 'image'

# ログ出力設定。ファイル&コンソール出力
LOG_FILE = OUT_DIR + os.path.sep + 'makeEdge.log' 
logging.basicConfig(filename=LOG_FILE, format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger()
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter('%(relativeCreated)s:%(levelname)s:%(funcName)s:%(message)s'))
logger.addHandler(log_handler)

def resize_image(src, dst_w, dst_h=0):
  w, h = src.shape[:2]
  if dst_h <= 0:
    dst_h = int(w/h * dst_w)
  return cv2.resize(src, (dst_w, dst_h))

def get_gray(src):
  return cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)


def make_canny(img_g):
  pass

def make_laplacian(img_g):
  laplacian = cv2.Laplacian(img_g, cv2.CV_64F)
  print('max:{0}, min:{1}'.format(np.max(laplacian), np.min(laplacian)))
  #dst = cv2.convertScaleAbs(laplacian)  # 正規化(0,255)
  dst = np.absolute(laplacian).astype(np.uint8)  # 正規化(0,255)
  print('max:{0}, min:{1}'.format(np.max(dst), np.min(dst)))
  return dst

def test02():
  img_path = INPUT_DIR + os.path.sep + 'sudoku.png'
  img_g = cv2.imread(img_path, 0)
  laplacian = make_laplacian(img_g)
  cv2.imshow('src', img_g)
  cv2.imshow('dst', laplacian)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def make_sobel(img_g):
  kernel_x = np.array([ [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
  kernel_y = np.array([ [ 1, 2, 1],
                [ 0, 0, 0],
                [-1,-2,-1]])

  gray_x = cv2.filter2D(img_g, cv2.CV_64F, kernel_x)
  gray_y = cv2.filter2D(img_g, cv2.CV_64F, kernel_y)
  dst = np.sqrt(gray_x **2 + gray_y**2)
  dst = cv2.convertScaleAbs(dst)  # 正規化(0,255)
  return dst

def test01():
  img_path = INPUT_DIR + os.path.sep + 'seikimatu-001.jpg'
  img = cv2.imread(img_path)
  img_g = get_gray(img)
  img_s = make_sobel(img_g)
  img_r = resize_image(img, 800)
  img_sr = resize_image(img_s, 800)
  cv2.imshow('src', img_r)
  cv2.imshow('dst', img_sr)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  test02()


