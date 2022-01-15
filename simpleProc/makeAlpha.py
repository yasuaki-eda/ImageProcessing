'''
 αチャンネルを持つ画像を作成します。
'''
import cv2
import numpy as np
import os
import math

OUT_DIR = '..' + os.path.sep + 'out'


'''
  画像サイズ(128, 64), 上半分に色を塗り、右半分をα=255(透過)にする。
'''
def test01():
  w = 128
  h = 64
  img = np.zeros((h, w, 4), dtype='uint8')
  img[:int(h/2), :, 2] = 255
  img[:,:int(w/2),3] = 255
  cv2.imwrite(OUT_DIR+os.path.sep +'alpha.png', img)
  return img

'''
  αチャンネルテスト2
  画像の読み込み
'''
def test02():
  img = cv2.imread(OUT_DIR+os.path.sep+'alpha.png', cv2.IMREAD_UNCHANGED)
  print(img.shape)

'''
  αチャンネルテスト3
  1. 画像サイズ(128, 64), 上半分を赤、下半分を青にする。
  2. α値を設定。左から0→255にグラデーションさせる 
'''
def test03():
  w = 128
  h = 64
  img = np.zeros((h, w, 4), dtype='uint8')
  img[:int(h/2),:,0] = 255
  img[int(h/2):,:,2] = 255

  # α値の設定
  y = np.tile(range(h), w).reshape(w, h).T
  x = np.tile(range(w), h).reshape(h, w)
  img[:, :, 3] = (x / w * 255).astype(np.uint8)

  # 結果の保存
#  cv2.imwrite(OUT_DIR+os.path.sep + 'alpha_test03.png', img)
  
  cv2.imshow('src', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

'''
  ブログに載せる用のコード
  test03を流用
'''
def blog01():
  w, h = 128, 64
  img = np.zeros((h, w, 4), dtype='uint8')
  img[:int(h/2),:,0] = 255
  img[int(h/2):,:,2] = 255

  # α値の設定
  x = np.tile(range(w), h).reshape(h, w)
  img[:, :, 3] = (x / w * 255).astype(np.uint8)
  cv2.imwrite('alpha1.png', img)
  return img

'''
  αブレンドの確認
  1. 背景を赤、前景を青にする。前景だけαチャンネルを設定してブレンドしてみる
  2. 背景を赤、前景を青にする。前景、背景にαチャンネルを設定してブレンドしてみる
  3. 背景をlena, 前景をグレーにする。lenaの顔周辺の矩形領域を透過、周辺をグレーにする。
  4. 背景をlena, 前景をbaboonにする。格子&グラデーションのαチャンネルを設定してブレンドしてみる
  5. かっこいいαブレンドをしてみる?
'''
# まずは前景だけαチャンネルを設定
def alpha1():
  w, h = 128, 64
  img_b = np.zeros((h, w, 4), dtype='float32')
  img_f = np.zeros((h, w, 4), dtype='float32')
  out = np.zeros((h, w, 4), dtype='float32')
  img_b[:,:,2] = 1
  img_f[:,:,0] = 1
  img_b[:,:,3] = 1
  img_f[:,:int(w/2),3] = 0.25

  out[:,:,3] = img_f[:,:,3] + (1-img_f[:,:,3])*img_b[:,:,3]
  for c in range(3):
    out[:,:,c] = np.where(out[:,:,3]==0, 0, (img_f[:,:,c]*img_f[:,:,3]+img_b[:,:,c]*(1-img_f[:,:,3])*img_b[:,:,3]))/out[:,:,3]
  
  cv2.imwrite('./alpha1.png', np.clip(out*255, 0, 255).astype(np.uint8))
  cv2.imshow('out', out)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def alpha2():
  w, h = 128, 64
  img_b = np.zeros((h, w, 4), dtype='float32')
  img_f = np.zeros((h, w, 4), dtype='float32')
  img_b[:,:,2] = 1
  img_f[:,:,0] = 1
  img_b[:int(h/2),:,3] = 0.75
  img_f[:,:int(w/2),3] = 0.25

  out = alpha_blend(img_f, img_b)
  cv2.imwrite('./alpha2.png', np.clip(out*255, 0, 255).astype(np.uint8))
  cv2.imshow('out', out)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



def alpha_blend(f, b):
  out = np.zeros_like(f, dtype='float32')
  out[:,:,3] = f[:,:,3] + (1-f[:,:,3])*b[:,:,3]
  for c in range(3):
    out[:,:,c] = np.where(out[:,:,3]==0, 0, (f[:,:,c]*f[:,:,3]+b[:,:,c]*(1-f[:,:,3])*b[:,:,3])/out[:,:,3])
  return out

'''
 @param f: 前景. 4ch, uint8
 @param b: 背景. 4ch, uint8
 @return : 4ch, uint8 αブレンド結果
'''
def alpha_blend2(f, b):
  out = np.zeros_like(f, dtype='float32')
  h, w = out.shape[:2]
  f_a = np.zeros((h, w, 1), dtype='float32')
  b_a = np.zeros((h, w, 1), dtype='float32')
  f_a = f[:,:,3] / 255
  b_a = b[:,:,3] / 255
  out[:,:,3] = (f_a + (1-f_a)*b_a) * 255
  for c in range(3):
    out[:,:,c] = np.where(out[:,:,3]==0, 0, ((f[:,:,c]*f_a+b[:,:,c]*(1-f_a)*b_a)/out[:,:,3]) )
  out2 = np.clip(out*255, 0, 255)
  return out2.astype(np.uint8)


def make_front_img1(img_b):
  h, w = img_b.shape[:2]
  img_f = np.zeros_like(img_b)
  ch, cw = int(h/2), int(w/2)
  x = np.tile(range(w), h).reshape(h, w)
  y = np.tile(range(h), w).reshape(w, h).T
  a = np.zeros((w, h), dtype='uint8')
  r = math.sqrt(ch**2 + cw**2)
  a = ((x-cw)**2 + (y-ch) **2) / r**2
  img_f[:,:,0] = 0
  img_f[:,:,1] = 0
  img_f[:,:,2] = 0
  img_f[:,:,3] = a*255
  return img_f

def make_front_img2(img_b):
  h, w = img_b.shape[:2]
  img_f = np.zeros_like(img_b)
  ch, cw = int(h/2), int(w/2)
  x = np.tile(range(w), h).reshape(h, w)
  y = np.tile(range(h), w).reshape(w, h).T
  a = np.zeros((w, h), dtype='uint8')
  r = math.sqrt(ch**2 + cw**2)
  a = ((x-cw)**2 + (y-ch) **2) / r**2
  img_f[:,:,0] = 0
  img_f[:,:,1] = 128
  img_f[:,:,2] = 0
  img_f[:,:,3] = a*255
  return img_f

# 前景を画像中心を明るく縁を暗くする
def test05():
  path_bg = '..' + os.path.sep + 'image' + os.path.sep + 'alpha_test03.png'
  img_src = cv2.imread(path_bg)
  h, w  = img_src.shape[:2]
  img_b = np.zeros((h,w,4), dtype='uint8')
  for c in range(3):
    img_b[:,:,c] = img_src[:,:,c]
  img_b[:,:,3] = 64
  img_f = make_front_img1(img_b)
  out = alpha_blend2(img_f, img_b)
  out2 = np.zeros((h, w, 3), dtype='uint8')
  for c in range(3):
    out2[:,:,c] = out[:,:,c]

  cv2.imshow('out', out)
  cv2.imwrite('..' + os.path.sep + 'out' + os.path.sep + 'test05.png', out2.astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# 前景を画像中心を明るく縁を暗くする
# 背景のα値を横方向にグラデーションさせる
def test06():
  path_bg = '..' + os.path.sep + 'image' + os.path.sep + 'alpha_test03.png'
  img_src = cv2.imread(path_bg)
  h, w  = img_src.shape[:2]
  img_b = np.zeros((h,w,4), dtype='uint8')
  for c in range(3):
    img_b[:,:,c] = img_src[:,:,c]
  # ↓↓ この処理を修正 ↓↓
  img_b[:,:,3] = 128

  img_f = make_front_img2(img_b)
  out = alpha_blend2(img_f, img_b)
  out2 = np.zeros((h, w, 3), dtype='uint8')
  for c in range(3):
    out2[:,:,c] = out[:,:,c]

  cv2.imshow('out', out)
  cv2.imshow('front', img_f)
  cv2.imwrite('..' + os.path.sep + 'out' + os.path.sep + 'alpha_test06_out.png', out2.astype(np.uint8))
  cv2.imwrite('..' + os.path.sep + 'out' + os.path.sep + 'alpha_test06_f.png', img_f.astype(np.uint8))
  cv2.imwrite('..' + os.path.sep + 'out' + os.path.sep + 'alpha_test06_b.png', img_b.astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def debug_test05():
  w, h = 16, 32
  img_f = np.zeros((h, w, 4), dtype='uint8')
  img_b = np.zeros((h, w, 4), dtype='uint8')

  img_f[:,:int(w/3),3] = 0
  img_f[:,int(w/3):int(w*2/3),3] = 128
  img_f[:,int(w*2/3):,3] = 255
  img_f[:,:,1] = 255
  img_f[:,:,3] = 128
  img_b[:,:,0] = 255
  img_b[:int(h/3),:,3] = 0
  img_b[int(h/3):int(h*2/3),:,3] = 128
  img_b[int(h*2/3):,:,3] = 255
  out =  alpha_blend2(img_f, img_b)
  print(out.shape)



'''
  αチャンネルテスト4
  ・αブレンドしてみる
  1. 

'''
def test04():
  img_b = cv2.imread('../image/lena.jpg')
  img_f = cv2.imread('../image/baboon.jpg')
  h, w= img_b.shape[:2]
  imgb = np.zeros((h, w, 4), dtype='uint8')
  imgf = np.zeros((h, w, 4), dtype='uint8')
  dst = np.zeros((h, w, 4), dtype='uint8')
  x = np.tile(range(w), h).reshape(h, w)
  y = np.tile(range(h), w).reshape(w, h).T
  PATTERN_SIZE = 32
  B_ALPHA = 50
  F_ALPHA = 200
  imgf[:,:,3] = np.where( ((x//PATTERN_SIZE)%2 == 0) & ((y//PATTERN_SIZE)%2 == 0), F_ALPHA, 0)
  imgf[:,:,3] = np.where( ((x//PATTERN_SIZE)%2 == 1) & ((y//PATTERN_SIZE)%2 == 1), F_ALPHA, imgf[:,:,2])
  for c in range(3):
    imgb[:,:,c] = img_b[:,:,c]
    imgf[:,:,c] = img_f[:,:,c]

  cv2.imshow('imgf', imgf)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
  test06()



