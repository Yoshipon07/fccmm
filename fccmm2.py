# coding: utf-8
# Your code here!
import numpy as np
import itertools as it
import time

CLUSTER = 3  # クラスター数
OBJECT = 10  # 個体数
ITEM = 8  # 項目数
ROOP = 5  # 実験繰り返し回数
LAMBDA = 1.0  # 個体メンバシップのファジィ度
HANTEI = 0.00001  # 収束判定用
MAX_CAL = 200  # 最大繰り返し回数
INPUT_FILENAME = "test_fccmm.csv"  # 読み込むファイル名(共起関係データ)
INPUT_ANSFILE = "test_ans.csv"  # 読み込むファイル名(正解クラスデータ)
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


# Rand-Indexの算出
def rand_index(u, label_data):
    max_cluster = np.argmax(u, axis=0)  # 各個体の最大メンバシップ値を持つクラスター番号(インデックス)を取得
    crisp_u = np.zeros((CLUSTER, OBJECT))  # 個体のクリスプ割り当て用の行列を初期化
    ans_u = np.zeros((CLUSTER, OBJECT))  # 個体の正解ラベル割り当て用の行列を初期化
    permutation = np.arange(CLUSTER)  # 順列生成用のリストを初期化

    table = np.zeros((CLUSTER, CLUSTER))  # クロス集計表の初期化
    tp = np.zeros((CLUSTER))  # True Positive(正しくpositiveに予測した数):正しく真のクラスに分類できている
    fp = np.zeros((CLUSTER))  # False Positive(間違ってpositiveに予測した数):偽のクラスが真のクラスに分類されている
    tn = np.zeros((CLUSTER))  # True Negative(正しくnegativeに予測した数):正しく偽のクラスに分類できている
    fn = np.zeros((CLUSTER))  # False Negative(間違ってnegativeに予測した数):真のクラスが偽のクラスに分類されている
    pre = np.zeros((CLUSTER))  # Precision(適合率):正と予測したデータのうち，実際に正であるものの割合
    rec = np.zeros((CLUSTER))  # Recall(感度):実際に正であるもののうち，正であると予測されたものの割合
    f_macro = np.zeros((CLUSTER))  # F値:正解率と再現率の調和平均
    ri = np.zeros((CLUSTER))  # Rand-Index 正解クラスと得られたクラスとの一致率

    for i in range(OBJECT):
        crisp_u[max_cluster[i]][i] = 1  # 個体をクリスプに割り当てる(1なら帰属、0なら帰属しない)
        ans_u[label_data[i]][i] = 1  # 個体の正解ラベルを割り当てた行列を生成

    # 正解クラスとの一致率が最大のクラスター番号の組み合わせを採用(順列を用いて)
    eval_max = 0.0  # 最大評価値を0に初期化
    for p in it.permutations(permutation):
        count = 0
        for c in range(CLUSTER):
            for i in range(OBJECT):
                if crisp_u[p[c]][i] == 1 and ans_u[c][i] == 1:  # 正解クラスと一致すればカウンターをインクリメント
                    count += 1
        eval = count / OBJECT  # 評価値を算出
        if eval > eval_max:  # 最大の評価値(一致率)が更新された場合、その時のクラスター番号の組み合わせを格納
            eval_max = eval
            index = p

    # クロス集計表の作成(行：正解クラス * 列：得られたクラス)
    for i in range(OBJECT):
        for c in range(CLUSTER):
            if crisp_u[index[c]][i] == 1:
                table[label_data[i]][c] += 1

    # クロス集計表のイメージ
    #             　　　　　　       　　　得られたクラス
    # 　　　　　　          　　　 　真                          偽
    # 正解クラス　　　　　真　　True Positive　　　　　　　　False Negative
    # 　　　　　　　　　　偽　　False Positive　　　　　　　 True Negative

    for c in range(CLUSTER):
        tp[c] = table[c][c]  # True Positiveを格納
        for i in range(CLUSTER):
            fp[c] += table[i][c]  # False Positiveのための計算
            fn[c] += table[c][i]  # False Negativeのための計算
            pre[c] += table[i][c]  # Precisionのための計算
            rec[c] += table[c][i]  # Recallの為の計算
        fp[c] -= tp[c]  # True Positiveを引いて完成
        fn[c] -= tp[c]  # True Positiveを引いて完成
        tn[c] = OBJECT - (tp[c] + fn[c] + fp[c])  # True Negativeの完成
        ri[c] = (tp[c] + tn[c]) / OBJECT  # Rand-Indexの完成
        if pre[c] == 0.0:
            pre[c] = 0.0
        else:
            pre[c] = tp[c] / pre[c]  # Precisionの完成
        if rec[c] == 0.0:
            rec[c] = 0.0
        else:
            rec[c] = tp[c] / rec[c]  # Recallの完成
        if pre[c] == 0.0 and rec[c] == 0.0:
            f_macro[c] = 0.0
        else:
            f_macro[c] = 2.0 * pre[c] * rec[c] / (pre[c] + rec[c])  # F値の算出

    # 以下、出力用のnumpy行列を作成
    list_above = [""]  # 左上に空白を作成するため
    list_above_parts = ["TP", "FP", "TN", "FN", "PRE(適合率)", "REC(再現率)", "F_MACRO(F値)", "Rand-Index"]  # 各値の名前を格納
    list_above = np.append(list_above, ["クラスター" + str(i + 1) for i in range(CLUSTER)])  # クラスターの名前を追加
    list_above = np.append(list_above, list_above_parts)  # 二つのリストを結合(出力用テーブルの上部完成)

    list_side = ["クラス" + str(i + 1) for i in range(CLUSTER)]  # クラスの名前を格納
    list_side = np.reshape(list_side, (CLUSTER, 1))  # リストを結合するために変形させる(1列に変形)

    output_table = np.stack((tp, fp, tn, fn, pre, rec, f_macro, ri), axis=1)  # 各値を格納したリストを結合させる(1次元リストから2次元リストへ)
    output_table = np.concatenate((table, output_table), axis=1)  # 各値を格納したリストを結合させる(2次元同士の結合)
    output_table = np.hstack((list_side, output_table))  # リストの結合(1次元と2次元のリストを横にくっつける)
    output_table = np.vstack((list_above, output_table))  # リストの結合(1次元と2次元のリストを縦にくっつける) 出力用のテーブル完成

    return output_table  # 出力用のテーブルを返す


# ファイルの書き込み
def output(u, w, output_table):
    with open(OUTPUT_FILENAME, "a") as o:
        np.savetxt(o, u, delimiter=",", fmt="%.8f")
        o.write("\n")
        np.savetxt(o, w, delimiter=",", fmt="%.8f")
        o.write("\n")
        np.savetxt(o, output_table, delimiter=",", fmt="%.8s")  # 文字列も含まれるためフォーマットを"%.8s"にしている
        o.write("\n")


r = np.loadtxt(INPUT_FILENAME, delimiter=",")  # ファイルの読み込み
label_data = np.loadtxt(INPUT_ANSFILE, delimiter=",", dtype="int8")  # ファイルの読み込み(正解クラスデータ)

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

    output_table = rand_index(u, label_data)  # Rand-Indexなどの値を算出し、出力用のテーブルを作成
    output(u, w, output_table)  # ファイルの書き込み
    # 以下メンバシップを出力(動作確認)
    print(u)
    print()
    print(w)
    print(flag_u)
    print(flag_w)
    print(t)