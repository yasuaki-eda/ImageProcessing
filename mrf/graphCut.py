import cv2
import numpy as np
import sys
sys.path.append('../baseproc')
import quantization
import noise
import queue
import time

'''
  E(X) = Σgv(Xv) + Σh(Xu, Xv)
  gv(Xv) = A|Xv-Yv| , Yvは観測データ
  h(Xu,Xv) = B|Xu-Xv| u,v∈Vn (隣接項)
'''
class MRF_GC:
  '''
    @param src: 二値(0,1)入力画像(ノイズあり) 
    @param A : 入力値からの変化に関する係数
    @param B : 隣接項との変化に関する係数
  '''
  def __init__(self, src, A, B):
    self.src = src
    self.A = A
    self.B = B
    self.H, self.W = src.shape
    self.Ch = 6
    self.C = self.init_cost2d(self.src, self.A, self.B, self.Ch)
    self.G = self.init_graph2d(self.H, self.W, self.Ch)

    print(self.C[:,:,0])
    print(self.C[:,:,1])
    print(self.C[:,:,2])
    print(self.C[:,:,3])
    print(self.C[:,:,4])
    print(self.C[:,:,5])
    print(self.G[:,:,0])
    print(self.G[:,:,1])
    print(self.G[:,:,2])
    print(self.G[:,:,3])
    print(self.G[:,:,4])
    print(self.G[:,:,5])
#    print(self.C.shape, self.G.shape)

  def max_flow(self):
    Cf = np.copy(self.C)
    G1 = np.copy(self.G)
    h, w, ch = G1.shape
    S = 0
    T = h*w+1
    it = 0

    while True:
      it += 1
      d, pi = BFS_2d(G1)
      if pi[T] < 0:
        break

      # tへのaugmenting pathを取得
      ap = []
      ap.append(T)
      x = int(T)
      while x > 0:
        x = int(pi[x])
        ap.append(x)
      ap.reverse()

      # augmenting path上の最小costを計算
      vx = (ap[1]-1) // w
      vy = (ap[1]-1) % w
      down_cost = Cf[vx, vy, 0]
#      print(ap, Cf[vx, vy,:])
      for p in range(2, len(ap)):
        p0 = ap[p-1]
        p1 = ap[p]
        vx = (p0-1) // w
        vy = (p0-1) % w
        if p1 == T:
          if 0 < Cf[vx, vy, 5] and Cf[vx, vy, 5] < down_cost:
            down_cost = Cf[vx, vy, 5]
            #print(p0, p1, vx, vy, 'down 5', down_cost)
        elif p1 - p0 == 1:
          if 0 < Cf[vx, vy, 1] and Cf[vx, vy, 1] < down_cost:
            down_cost = Cf[vx, vy, 1]
            print(it, p0, p1, vx, vy, 'down 1', down_cost)
        elif p1 - p0 == -1:
          if 0 < Cf[vx, vy, 3] and Cf[vx, vy, 3] < down_cost:
            down_cost = Cf[vx, vy, 3]
            print(it, p0, p1, vx, vy, 'down 3', down_cost)
        elif p1 - p0 == w:
          if 0 < Cf[vx, vy, 2] and Cf[vx, vy, 2] < down_cost:
            down_cost = Cf[vx, vy, 2]
            print(it, p0, p1, vx, vy, 'down 2', down_cost)
        elif p1 - p0 == -w :
          if 0 < Cf[vx, vy, 4] and Cf[vx, vy, 4] < down_cost:
            down_cost = Cf[vx, vy, 4]
            print(it, p0, p1, vx, vy, 'down 4', down_cost)
     
#      print('down_cost', down_cost)

      # augmenting pathに沿ってflowを増加
      psize = len(ap)
      vx = (ap[1]-1) // w
      vy = (ap[1]-1) % w
      Cf[vx, vy, 0] -= down_cost
      for p in range(2, len(ap)):
        p0 = ap[p-1]
        p1 = ap[p]
        vx = (p0-1) // w
        vy = (p0-1) % w
        if p1 == T:
          Cf[vx, vy, 5] -= down_cost
          continue
        if p1 - p0 == 1:
          Cf[vx, vy, 1] -= down_cost
        elif p1 - p0 == -1:
          Cf[vx, vy, 3] -= down_cost
        elif p1 - p0 == w:
          Cf[vx, vy, 2] -= down_cost
        elif p1 - p0 == -w :
          Cf[vx, vy, 4] -= down_cost
        else:
          print('Augmenting Path error. p0', p0, ' p1:', p1)
          continue
#      print(ap, vx, vy, Cf[vx, vy,:])

      # G1を更新
      G1 = np.where(Cf<=0, -1, G1)
    
    F = self.C - Cf
#    print(self.C)
#    print(Cf)
#    print(F)
    return F, Cf, G1

  def init_cost2d(self, src, A, B, ch):
    h, w = src.shape
#    C = np.ones((h, w, ch)) * 10000
    C = np.zeros((h, w, ch))
    C[:,:-1,1] = B * np.abs(src[:,:-1]-src[:,1:])
    C[1:,:, 2] = B * np.abs(src[1:,:]-src[:-1,:])
    C[:,1: ,3] = B * np.abs(src[:,1:]-src[:,:-1])
    C[:-1,:,4] = B * np.abs(src[:-1,:]-src[1:,:])
    C[:,:,0] = A * np.abs(src[:,:]-1) + np.sum(C[:,:,1:-1], axis=2)
    C[:,:,5] = A * np.abs(src[:,:]-0) + np.sum(C[:,:,1:-1], axis=2)
    return C
  
  def init_graph2d(self, row, col, ch):
    G = np.zeros((row, col, ch), dtype='int32')
    h, w, c = G.shape
    G[:,:,5] =  h*w+1
    G[:,:,0] = np.array(range(1,h*w+1)).reshape(h,w)
    G[:,:,1] = np.array(range(2,h*w+2)).reshape(h,w)
    G[:,-1,1] = -1
    G[:,:,2] = np.array(range(-(w-1),h*w-(w-1))).reshape(h,w)
    G[0,:,2] = -1
    G[:,:,3] = np.array(range(h*w)).reshape(h,w)
    G[:,0,3] = -1
    G[:,:,4] = np.array(range(w+1, h*w+w+1)).reshape(h,w)
    G[-1,:,4] = -1
    return G

    
'''
# Breadth First Search
 @param G:探索対象の2dグラフ(G[:,:,0]=nlink(StoV), G[:,:,1:5]=tlink(VtoV), G[:,:,5]=nlink(VtoT))
 @return d:sからの距離1次元データ
 @return pi:parent要素番号1次元データ
'''
def BFS_2d(G):
  h, w, c = G.shape
  q = queue.Queue()
  q0 = queue.Queue()
  WHITE, GRAY, BLACK = 0, 1, 2
  size = h * w + 2  # {V,s,t}
  S = 0
  T = h*w+1
  color = np.ones(size)*WHITE
  d = np.ones(size) * 1e10
  pi = np.ones(size) * (-1)
  color[S] = GRAY
  d[S] = 0
  q.put(G[:,:,0].reshape(-1))
  q0.put(S)
  while not q.empty():
    u = q.get()
    u0 = q0.get()
    for v in u:
      if v < 0:
        continue
      if color[v] == WHITE:
        color[v] = GRAY
        d[v] = d[u0]+1
        pi[v] = u0
        if v != S and v != T:
          q.put(G.reshape(-1,c)[v-1,1:])
          q0.put(v)
    color[u0] = BLACK
  return d, pi




'''
 s, t, 2Dのvertex(画像)からなるGraphを作成します。
 vertexは隣接4項とtlinkを持ち、s, tは全vertexとlink(nlink)を持ちます。
 G[0]:s(source)からvertexへ
 G[1]-G[4]:vertexのtlink
 G[5]:vertexからt(sink)へ
'''
def init_graph2d(row, col):
  G = np.zeros((row, col, 6), dtype='int32')
  h, w, c = G.shape
  G[:,:,5] =  h*w+1
  G[:,:,0] = np.array(range(1,h*w+1)).reshape(h,w)
  G[:,:,1] = np.array(range(2,h*w+2)).reshape(h,w)
  G[:,-1,1] = -1
  G[:,:,2] = np.array(range(-(w-1),h*w-(w-1))).reshape(h,w)
  G[0,:,2] = -1
  G[:,:,3] = np.array(range(h*w)).reshape(h,w)
  G[:,0,3] = -1
  G[:,:,4] = np.array(range(w+1, h*w+w+1)).reshape(h,w)
  G[-1,:,4] = -1
  print(G[:,:,0])
  print(G[:,:,1])
  print(G[:,:,2])
  print(G[:,:,3])
  print(G[:,:,4])
  print(G[:,:,5])
  return G


def test_BFS2d():
  G = init_graph2d(10, 12)
  d, pi = BFS_2d(G)
  h, w, c = G.shape
  print(d[1:h*w+1].reshape(h,w))
  print(pi[1:h*w+1].reshape(h,w))
  print(d)
  print(pi)

def test_MRF_GC2():
  img = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
  ])
  gc = MRF_GC(img, 10, 15)
  F, Cf, G1 = gc.max_flow()
  dst = np.where(Cf[:,:,0]<=0, 255, 0)
  for c in range(F.shape[2]):
    print('F['+str(c)+']---------------------')
    print(F[:,:,c])
  for c in range(Cf.shape[2]):
    print('Cf['+str(c)+']---------------------')
    print(Cf[:,:,c])
  for c in range(G1.shape[2]):
    print('G1['+str(c)+']---------------------')
    print(G1[:,:,c])
  if dst is not None:
    cv2.imshow('dst', dst.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_MRF_GC():
  img = cv2.imread('../image/imori.jpg')
  img_gb = quantization.binarization(img)
  img_gbn = noise.make_noise(img_gb)
  img_gbn = np.where(img_gbn>0, 1, 0)
  t0 = time.time()

  gc = MRF_GC(img_gbn, 10, 15)
  F, Cf, G1 = gc.max_flow()
  for c in range(F.shape[2]):
    np.savetxt('./F_' + str(c) + '.csv', F[:,:,c], fmt='%d')
  for c in range(G1.shape[2]):
    np.savetxt('./G_' + str(c) + '.csv', G1[:,:,c], fmt='%d')
  for c in range(Cf.shape[2]):
    np.savetxt('./Cf_' + str(c) + '.csv', Cf[:,:,c], fmt='%d')
  dst = np.where(Cf[:,:,0]<=0, 255, 0)
  t1 = time.time()
  print('time:', t1-t0, '[s]')
  if dst is not None:
    cv2.imshow('n', (dst).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./imori_gc.jpg', dst.astype(np.uint8))
    cv2.imwrite('./imori_noise.jpg', (img_gbn*255).astype(np.uint8))

def test_MRF_GC_result():
#  F0 = np.loadtxt('./F_0.csv')
#  C0 = np.loadtxt('./C_0.csv')
#  img = np.where(F0==C0, 255, 0)
  Cf = np.loadtxt('./Cf_0.csv')
  img = np.where(Cf<=0, 255, 0)
  cv2.imshow('res', img.astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__':
#  test_BFS2d()
  test_MRF_GC2()
#  test_MRF_GC_result()

