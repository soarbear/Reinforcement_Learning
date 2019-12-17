#
# Title: The best policy calculated with MDP value itertation method
# Author: T.Yanagi
#
import numpy as np
import copy
#
# 価値反復法で3状態x2行動のMDPマルコフ決定過程を解く
#
def value_iteration():

  # 行動の条件確率p[s(t+1)|s(t),a(t)]の設定
  p = [0.5, 0.5, 0.5]

  # 割引率γの設定
  gamma = 0.90
  
  # ループ終了判定基準係数の設定
  epsilon = 0.001

  # 報酬値r[s]の設定
  r = [0.5, 0.5, 0.5, 1.0, 0.0]
  
  # 価値関数r[s]の初期化
  v = [0.0, 0.0, 0.0, 0.0, 0.0]
  v_new = copy.copy(v)

  # 状態-行動価値関数q(s,a)の初期化
  q = np.zeros((3, 2))

  # 方策分布(pi(s))の初期化
  pi = [0.5, 0.5, 0.5]

  # 状態s(t+1)の初期化 3x2
  s_t1 = np.zeros((3, 2))
  s_t1[0, 0] = 2 # up
  s_t1[0, 1] = 1 # right
  s_t1[1, 0] = 3 # up
  s_t1[1, 1] = 4 # right(out)
  s_t1[2, 0] = 4 # up(out)
  s_t1[2, 1] = 3 # right

  iteration = 0
  # 価値反復法の計算
  while True:
    delta = 0.0
    for i in range(3):

      # 行動価値関数を計算
      q[i, 0] = p[i] * (r[int(s_t1[i,0])] + gamma * v[int(s_t1[i,0])])
      q[i, 1] = (1.0 - p[i]) * (r[int(s_t1[i,1])] + gamma * v[int(s_t1[i,1])])

      # 行動価値関数のもとで greedy に方策を改善
      if q[i, 0] > q[i, 1]:
        pi[i] = 1
      elif q[i, 0] == q[i, 1]:
        pi[i] = 0.5
      else:
        pi[i] = 0

    # 改善された方策のもとで価値関数を計算
    v_new = np.append(np.max(q, axis=-1), [0.0, 0.0])

    # 現ステップの価値関数と方策を表示
    print(f'iteration: {iteration}, value: {v_new[:3]} policy: {pi}')
    
    delta = max(delta, abs(np.min(v_new[:3] - v[:3])))
    # 計算された価値関数 v_new が前ステップの値 v を改善しなければ終了
    if delta <= epsilon * (1 - gamma) / gamma:
      # 最適方策を返す
      return pi
    
    # 価値関数を更新
    v = copy.copy(v_new)
    iteration += 1
  
#
# main()
#
if __name__ == '__main__':
  
  # 価値反復法最適方策を探る
  pi = value_iteration()

  # 行動方向表示の初期化
  direction = ['','','']

  # 最適方策を表示
  for i in range(3):
    if pi[i] == 1.0:
      direction[i] = '^'
    elif pi[i] == 0.5:
      direction[i] = 'L'
    else:
      direction[i] = '>'
  print(f'{direction[2]} 0\n{direction[0]} {direction[1]}')
