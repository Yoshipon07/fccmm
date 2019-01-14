# coding: utf-8
# Your code here!
import numpy as np
import math
import time

CLUSTER = 3  # クラスター数
OBJECT = 10  # 個体数
ITEM = 8  # 項目数
ROOP = 10  # アルゴリズム繰り返し回数
BETA_MAX = 10.0  # 項目の排他性を高める度合
LAMBDA = 1.0  # 個体メンバシップのファジィ度
HANTEI = 0.00001  # 収束判定用
INPUT_FILENAME = "test_fccmm.csv"  # 読み込むファイル名
OUTPUT_FILENAME = "test_output.csv"  # 出力するファイル名

#fccmm1 (fccmm法の改良版)
#fccmm法にペナルティ項Scjを付与し、項目の排他的分割を実現させる
#################################################
# 行列 = 行 * 列 = line * row
#
#
##################################################


# 個体メンバシップ初期化
def initialize_u():
    u = np.random.rand(CLUSTER, OBJECT)  # 乱数生成
    row_sum = np.sum(u, axis=0)  # 各個体の和(各列の和)を算出
    return u / row_sum  # 正規化(各個体の和が1)


# 項目メンバシップ初期化
def initialize_w():
    w = np.random.rand(CLUSTER, ITEM)  # 乱数生成
    line_sum = np.sum(w, axis=1)  # 各行の和を算出
    line_sum = np.reshape(line_sum, (CLUSTER, 1))  # 算出した和が1行となっているので、1列に変換(計算のため)
    return w / line_sum  # 正規化(各クラスターにおいて、項目の和が1)


# 共有ペナルティ初期化
def initialize_s():
    return np.ones((CLUSTER, ITEM))  # 全要素が1の配列を生成


# クラスター容量更新
def update_pai(u):
    return np.sum(u, axis=1) / OBJECT  # 各クラスターのメンバシップの平均(各行の平均)を算出


# 個体メンバシップの更新
def update_u(u, w, s, r, pai, LAMBDA):
    numerator_u = np.zeros((CLUSTER, OBJECT))  # numeratorの初期化(0で初期化)
    for i in range(OBJECT):
        for c in range(CLUSTER):
            numerator_u[c][i] = pai[c]  # numeratorにπを代入
            for j in range(ITEM):
                numerator_u[c][i] *= np.power(w[c][j], (r[i][j] * s[c][j] / LAMBDA))  # 累乗計算(ここで分子完成)
        denominator_u = np.sum(numerator_u, axis=0)  # 分子の総和(各列の和)を算出
        for c in range(CLUSTER):
            u[c][i] = numerator_u[c][i] / denominator_u[i]  # 個体メンバシップの更新


# 項目メンバシップの更新(C++チックな更新手順)
# def update_w2(u,w,r):
#     numerator_w = np.zeros((CLUSTER,ITEM))
#     for c in range(CLUSTER):
#         for j in range(ITEM):
#             for i in range(OBJECT):
#                 numerator_w[c][j] += r[i][j] * s[c][j] * u[c][i]
#         denominator_w = np.sum(numerator_w, axis = 1)
#         for j in range(ITEM):
#             w[c][j] = numerator_w[c][j] / denominator_w[c]

def update_w(u, w, s, r):
    numerator_w = np.dot(u, r)  # 行列uとrの積を計算
    for c in range(CLUSTER):
        for j in range(ITEM):
            numerator_w *= s[c][j]  # できた行列の積に共有ペナルティScjを乗算
    denominator_w = np.sum(numerator_w, axis=1)  # numeratorの各行の和を算出
    for c in range(CLUSTER):
        for j in range(ITEM):
            w[c][j] = numerator_w[c][j] / denominator_w[c]  # 項目メンバシップの更新


# 共有ペナルティ更新
def update_s(s, w):
    global t
    if t < BETA_MAX:
        beta = 0.1 * (t-1)
    else:
        beta = BETA_MAX
    sum = np.sum(w, axis=0)  #各列の和を格納
    for c in range(CLUSTER):
        for j in range(ITEM):
            s[c][j] = math.exp(-beta * (sum[j] - w[c][j]))  # クラスターc以外に帰属する項目メンバシップの和と-betaの積をexp関数に投げる
    t+=1


# 収束判定
def judge_convergent(u, pre_u, w, pre_w):
    global flag
    flag = 1
    for c in range(CLUSTER):
        for i in range(OBJECT):
            if np.abs(pre_u[c][i] - u[c][i]) > HANTEI:
                flag = 0
        for j in range(ITEM):
            if np.abs(pre_w[c][j] - w[c][j]) > HANTEI:
                flag = 0


# ファイルの書き込み
def output(u, w, s):
    with open(OUTPUT_FILENAME, "a") as o:
        o.write("\n")
        np.savetxt(o, u, delimiter=",", fmt="%.8f")
        o.write("\n")
        np.savetxt(o, w, delimiter=",", fmt="%.8f")
        o.write("\n")
        np.savetxt(o, s, delimiter=",", fmt="%.8f")


r = np.loadtxt(INPUT_FILENAME, delimiter=",")  # ファイルの読み込み

for roop in range(ROOP):
    np.random.seed(seed=roop)  # 乱数のシード値初期化
    u = initialize_u()  # 個体メンバシップの初期化
    w = initialize_w()  # 項目メンバシップの初期化
    s = initialize_s()  # 共有メンバシップの初期化
    flag = 0  # 終了フラグの初期化
    t = 1  # 反復回数の初期化
    while flag == 0:  # 収束フラグが1(収束)するまで繰り返す
        pre_u = u.copy()  # 更新前の個体メンバシップを保存
        pre_w = w.copy()  # 更新前の項目メンバシップを保存
        pai = update_pai(u)  # クラスター容量の更新
        update_u(u, w, s, r, pai, LAMBDA)  # 個体メンバシップの更新
        update_w(u, w, s, r)  # 項目メンバシップの更新
        update_s(s, w)  # 共有ペナルティの更新
        judge_convergent(u, pre_u, w, pre_w)  # 収束判定
    output(u, w, s)  # ファイルの書き込み
    print(u)
    print()
    print(w)
    print()
    print(s)