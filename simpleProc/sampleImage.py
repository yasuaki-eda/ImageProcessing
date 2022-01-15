import numpy as np
import cv2
import os.path
from PIL import Image


'''
 サンプル画像を作成します。
'''

IMAGE_BMP = 1
IMAGE_JPG = 2
IMAGE_PNG = 3
IMAGE_GIF = 4

def get_number_image(start, end, type, out_dir):

  FONT_SIZE = 4
  FONT_SCALE = 6
  COLOR = (255,255,255)

  for i in range(start, end+1):
    img = np.zeros((160, 120, 3)).astype(np.uint8)
    if type == IMAGE_JPG:
      img[:,:,1] = 128
    elif type == IMAGE_PNG:
      img[:,:,0] = 128
      img[:,:,2] = 128

    txt = "{:>02d}".format(i)
    cv2.putText(img, txt, (0, 110), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, COLOR, FONT_SIZE, 8)

    if i == start:
      if type == IMAGE_BMP:
        ext = '.bmp'
        out_dir += os.path.sep + 'bmp'
      elif type == IMAGE_JPG:
        ext = '.jpg'
        out_dir += os.path.sep + 'jpg'
      elif type == IMAGE_PNG:
        ext = '.png'
        out_dir += os.path.sep + 'png'
      else :
        return None
    cv2.imwrite(out_dir + os.path.sep + txt + ext, img)

  return None

def get_size_image():

  FONT_SIZE = 4
  FONT_SCALE = 6
  COLOR = (255,255,255)
  out_dir = '../out/size'

  size_pattern = [
    [5000, 100],
    [100, 5000],
    [1, 1],
    [5000, 5000]
  ]
  sp = np.array(size_pattern)
  for i in range(sp.shape[0]):
    img = np.ones((sp[i,0], sp[i,1], 3)).astype(np.uint8) * 128
    name = '(' + str(sp[i,0]) + '-' + str(sp[i,1]) + ')'
    cv2.imwrite(out_dir+os.path.sep+name+'.bmp', img)
    cv2.imwrite(out_dir+os.path.sep+name+'.jpg', img)
    cv2.imwrite(out_dir+os.path.sep+name+'.png', img)

  return None 


# ---------------------------------
# GIF画像を作成し保存します
# ---------------------------------
def get_gif_image():
  img = np.zeros((160, 120, 3)).astype(np.uint8)
  img2 = np.ones((160, 120, 3)).astype(np.uint8) * 128
  img3 = np.ones((160, 120, 3)).astype(np.uint8) * 255
  images = []
  images.append(Image.fromarray(img))
  images.append(Image.fromarray(img2))
  images.append(Image.fromarray(img3))
  images[0].save('../out/160-120.gif', save_all=True,append_images=images[1:], optimize=False, loop=0)


def get_0byte():
  img = np.zeros((0,0,3)).astype(np.uint8)
  cv2.imwrite('../out/0byte.bmp', img)
  #cv2.imwrite('../out/0byte.png', img)

def test_number_image():
  get_number_image(0, 99, IMAGE_BMP, '../out')
  return None


def test_number_image1():
  FONT_SIZE = 4
  FONT_SCALE = 6
  COLOR = (255,255,255)

  img = np.zeros((160, 120, 3)).astype(np.uint8)
  txt = "{:>02d}".format(76)
  cv2.putText(img, txt, (0, 110), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, COLOR, FONT_SIZE, 8)

  return img

def get_png_image():
  img = cv2.imread('../image/20210301173428_overLay.bmp')
  cv2.imwrite('../out/20210301173428_overLay.png', img)



if __name__ == '__main__':
  #img = get_size_image()
  img = None

  if img is not None:
    cv2.imshow('dst', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


