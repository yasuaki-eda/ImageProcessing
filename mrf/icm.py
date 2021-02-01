'''
 iterated conditional modesによるノイズ除去
'''
import cv2
import numpy as np
import sys
sys.path.append('../baseproc')
from gray import BGR2GRAY
from quantization import binarization
from noise import make_noise

'''
 ICMによる二値画像のノイズ除去を実施します。
  コンストラクタ
    @param img_bin:ノイズ画像, [-1, 1]で二値化
    @eta : tlinkの係数. 元画像と同じ値を取るときに低いエネルギーとなる
    @h : 画素値に対して一様に作用する係数.
    @beta : nlinkの係数. 周辺画素と同じ時に低いエネルギーとなる
  計算 exec_icm()
  結果の取得 get_dstimage()
'''
class MRF_ICM:
  def __init__(self, img_bin, eta=2.1, h=0, beta=1):
    self.src = img_bin
    self.H, self.W = self.src.shape
    self.dst = (-1)*np.ones((self.H+2, self.W+2))
    self.dst[1:self.H+1, 1:self.W+1] = img_bin[:,:]
    self.eta = eta
    self.h = h
    self.beta = beta
    self.energy = 0
    self.max_phase = 10000
  
  # ICMの実行
  def exec_icm(self):
    x = np.tile(range(self.H), self.W).reshape(self.H, self.W).T.reshape(-1)
    y = np.tile(range(self.W), self.H)
    rnum = self.H * self.W

    p = 0
    while p < self.max_phase:
      p += 1
      rlist = np.random.choice(range(rnum), rnum, replace=False)
      has_change = False
      for r in rlist:
        has_change += self.change_1px(x[r], y[r])
      print('change:', has_change)
      if has_change == 0:
        print('p:', p, ' break.')
        break
  
  # ICMの内部関数
  # energyが減少する場合、ラベルを反転させます
  def change_1px(self, x, y):
    vx = self.dst[x+1, y+1]
    vy = self.src[x, y]
    e0 = 0
    e0 += vx * self.h
    e0 -= vx * vy * self.eta
    e0 -= 2 * vx * self.beta * (self.dst[x, y+1] + self.dst[x+2, y+1] + self.dst[x+1, y] + self.dst[x+1, y+2])

    e1 = 0
    e1 += (-1) * vx * self.h
    e1 -= (-1) * vx * vy * self.eta
    e1 -= (-1) * 2 * vx * self.beta * (self.dst[x, y+1] + self.dst[x+2, y+1] + self.dst[x+1, y] + self.dst[x+1, y+2])

    has_change = False
    if e1 < e0:
      self.dst[x+1, y+1] *= -1
      has_change = True
    return has_change

  # 検算用に画像全体のエネルギーを計算
  def calc_energy(self, img):
    e = 0
    img4 = self.make_arround4(img)
    e -= self.beta * np.sum(img*img4[:,:,0])
    e -= self.beta * np.sum(img*img4[:,:,1])
    e -= self.beta * np.sum(img*img4[:,:,2])
    e -= self.beta * np.sum(img*img4[:,:,3])
    e += self.h * np.sum(img)
    e -= self.eta * np.sum(img * self.src)
    return e

  def make_arround4(self, img):
    h, w = img.shape
    res = np.zeros((h, w, 4))
    res[  :  ,  :-1,0] = img[:  ,1:]
    res[ 1:  ,  :  ,1] = img[:-1,:]
    res[  :  , 1:  ,2] = img[:  ,:-1]
    res[  :-1,  :  ,3] = img[1: ,:]
    return res

  # 計算結果画像の取得
  def get_dstimage(self):
    return self.dst[1:self.H+1, 1:self.W+1]


# [0,1]bin画像を [-1, +1]に変換します。  
def change_bin11(img):
  des = np.copy(img)
  return np.where(des==0, -1, 1)


def test_icm(img):
  img_g = BGR2GRAY(img)
  img_n = make_noise(img_g, rate=0.1)
  img_b = change_bin11(binarization(img_n))

  mrf = MRF_ICM(img_b)
  e = mrf.calc_energy(mrf.src)
  print('start energy:', e)
  mrf.exec_icm()
  res = mrf.get_dstimage()
  e = mrf.calc_energy(res)
  print('end energy:', e)

  # result check
  res = np.where(res==-1, 0, 255).astype(np.uint8)
  img_b = np.where(img_b==-1, 0, 255).astype(np.uint8)
  org_b = np.where(binarization(img_g)==0, 0, 255).astype(np.uint8)

  d1 = res[(np.abs(res - org_b) > 0)].shape[0]
  h, w = img_g.shape
  result_ratio = d1/(h*w)
  print('result error ratio:', result_ratio)

  # imshow
  cv2.imshow('dst', res)
  cv2.imshow('src', img_g)
  cv2.imshow('src_b', org_b)
  cv2.imshow('noise', img_b)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  img = cv2.imread('../image/imori.jpg')
  test_icm(img)



