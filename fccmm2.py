# coding: utf-8
# Your code here!
import numpy as np
import time

CLUSTER = 3  # クラスター数
OBJECT = 10  # 個体数
ITEM = 8  # 項目数
ROOP = 5  # 実験繰り返し回数
LAMBDA = 1.0  # 個体メンバシップのファジィ度
HANTEI = 0.00001  # 収束判定用
MAX_CAL = 200  # 最大繰り返し回数
INPUT_FILENAME = "test_fccmm.csv"  # 読み込むファイル名
OUTPUT_FILENAME = "test_output.csv"  # 出力するファイル名


#########################################################################
# 従来のfccmm法における項目の制約条件を変更した手法 (fccmm2)
# 二つの目的関数を用いてメンバシップを更新するヒューリスティックアプローチ
# 従来とは違う制約条件・目的関数を用いることで項目の排他的分割が実現される
#
# INPUT_FILENAMEに読み込みたいcsvファイルを指定
# それに対応したクラスター数・個体数・項目数を設定してあげる
#
# OUTPUT_FILENAMEに出力したいcsvファイルを指定
# 出力ファイルには個体のメンバシップ、一行空けて、項目のメンバシップが出力
#
# コードを読む際に混乱しない為、行列の整理
# 行列 = 行 * 列 = line * row　= axis(1) * axis(0)
#########################################################################


# 個体メンバシップ初期化
def initialize_u():
    u = np.random.rand(CLUSTER, OBJECT)  # 乱数生成
    row_sum = np.sum(u, axis=0)  # 各個体の和(各列の和)を算出
    return u / row_sum  # 正規化(各個体の和が1)


# 項目メンバシップ初期化
def initialize_w():
    w = np.random.rand(CLUSTER, ITEM)  # 乱数生成
    row_sum = np.sum(w, axis=0)  # 各項目の和(各列の和)を算出
    return w / row_sum  # 正規化(各項目の和が1)


# クラスター容量更新
def update_pai(u):
    return np.sum(u, axis=1) / OBJECT  # 各クラスターのメンバシップの平均(各行の平均)を算出


# 個体メンバシップの更新
def update_u(u, w, r, pai, LAMBDA):
    numerator_u = np.zeros((CLUSTER, OBJECT))  # numeratorの初期化(0で初期化)
    sum_w = np.sum(w, axis=1)  # 各クラスターにおける項目の和(各行の和)を算出
    for i in range(OBJECT):
        for c in range(CLUSTER):
            numerator_u[c][i] = pai[c]  # numeratorにπ(クラスター容量)を代入
            for j in range(ITEM):
                numerator_u[c][i] *= np.power(w[c][j] / sum_w[c], (r[i][j] / LAMBDA))  # 累乗計算(ここで分子完成)
    denominator_u = np.sum(numerator_u, axis=0)  # 分子の総和(各列の和)を算出(ここで分母完成)
    for c in range(CLUSTER):
        for i in range(OBJECT):
            u[c][i] = numerator_u[c][i] / denominator_u[i]  # 個体メンバシップの更新


# 項目メンバシップの更新(C++チックな更新手順)
# def update_w2(u,w,r):
#     numerator_w = np.zeros((CLUSTER,ITEM))
#     sum_u = np.sum(u, axis=1)
#     for j in range(ITEM):
#         for c in range(CLUSTER):
#             for i in range(OBJECT):
#                 numerator_w[c][j] += r[i][j] * u[c][i] / sum_u[c]
#     denominator_w = np.sum(numerator_w, axis = 0)
#     for c in range(CLUSTER):
#         for j in range(ITEM):
#             w[c][j] = numerator_w[c][j] / denominator_w[j]

# 項目メンバシップの更新
def update_w(u, w, r):
    sum_u = np.sum(u, axis=1)  # 各クラスターにおける個体の和(各行の和)を算出
    u_dash = np.zeros((CLUSTER, OBJECT))  # 再計算分を格納する配列を初期化
    for c in range(CLUSTER):
        for i in range(OBJECT):
            u_dash[c][i] = u[c][i] / sum_u[c]  # 各クラスターにおける個体の重要度を表現するように再計算
    numerator_w = np.dot(u_dash, r)  # 行列uとrの積を計算(ここで分子完成)
    denominator_w = np.sum(numerator_w, axis=0)  # numeratorの各列の和を算出(ここで分母完成)
    for c in range(CLUSTER):
        for j in range(ITEM):
            w[c][j] = numerator_w[c][j] / denominator_w[j]  # 項目メンバシップの更新


# 収束判定:収束閾値よりメンバシップの差が大きければフラグを0(未収束)に戻す
def judge_convergent(u, pre_u, w, pre_w):
    global flag_u, flag_w
    flag_u = 1
    flag_w = 1
    for c in range(CLUSTER):
        for i in range(OBJECT):
            if np.fabs(pre_u[c][i] - u[c][i]) > HANTEI:
                flag_u = 0
        for j in range(ITEM):
            if np.fabs(pre_w[c][j] - w[c][j]) > HANTEI:
                flag_w = 0


# ファイルの書き込み
def output(u, w):
    with open(OUTPUT_FILENAME, "a") as o:
        o.write("\n")
        np.savetxt(o, u, delimiter=",", fmt="%.8f")
        o.write("\n")
        np.savetxt(o, w, delimiter=",", fmt="%.8f")


r = np.loadtxt(INPUT_FILENAME, delimiter=",")  # ファイルの読み込み

# 先に設定した実験回数(ROOP)だけ繰り返す
for roop in range(ROOP):
    np.random.seed(seed=roop)  # 乱数のシード値初期化(シード値は実験回数に対応させている)
    u = initialize_u()  # 個体メンバシップの初期化
    w = initialize_w()  # 項目メンバシップの初期化
    flag_u = 0  # 個体収束フラグの初期化
    flag_w = 0  # 項目収束フラグの初期化
    t = 1  # アルゴリズム反復回数の初期化

    while flag_u == 0 and flag_w == 0 and t < MAX_CAL:  # 収束フラグが1(収束)する、または、最大繰り返し回数まで更新を繰り返す
        pre_u = u.copy()  # 更新前の個体メンバシップを保存
        pre_w = w.copy()  # 更新前の項目メンバシップを保存
        pai = update_pai(u)  # クラスター容量の更新
        update_u(u, w, r, pai, LAMBDA)  # 個体メンバシップの更新
        update_w(u, w, r)  # 項目メンバシップの更新
        judge_convergent(u, pre_u, w, pre_w)  # 収束判定
        t += 1  # アルゴリズム反復回数をインクリメント

    output(u, w)  # ファイルの書き込み
    # 以下メンバシップを出力(動作確認)
    print(u)
    print()
    print(w)
    print(flag_u)
    print(flag_w)
    print(t)
