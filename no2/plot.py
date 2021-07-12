import pickle
import time
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets.mnist import load_data
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt

def read_txt(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = float(lines[i].rstrip('\n'))
    return lines

def data_reshape(x):
    x = (x.reshape(len(x), 28 * 28)).astype(np.float64)
    x /= x.max()
    return x

def plot_data(data_list, data_name):
    activation_function_list = ['tanh', 'relu', 'logistic']
    train_rate = np.arange(0.1, 1.1, 0.1)
    plt.grid()
    plt.xlabel("Training Rate")
    plt.ylabel(data_name)
    plt.xticks(train_rate)
    for i, function_name in enumerate(activation_function_list):
        plt.plot(train_rate, data_list[i], marker='.', label=function_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)
    plt.show()

def heat_map(filename, x_test, y_label):
    import copy

    model = model_load(filename)
    start = time.time()
    predict = model.predict(x_test)
    duration = time.time() - start
    print("Time[s]: ", duration)
    
    labels = copy.deepcopy(y_label)
    labels = sorted(list(set(labels)))
    cmx_data = confusion_matrix(y_label, predict, labels=labels)
    print(cmx_data)
    print( classification_report(yt, predict) )
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True, fmt='2d')
    plt.show()

def model_load(filename):
    return pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    # 保存した識別率と実行時間を読み込み
    tanh_accuracies = read_txt("activation_function=tanh-accuracy.txt")
    tanh_times = read_txt("activation_function=tanh-time.txt")
    relu_accuracies = read_txt("activation_function=relu-accuracy.txt")
    relu_times = read_txt("activation_function=relu-time.txt")
    logistic_accuracies = read_txt("activation_function=logistic-accuracy.txt")
    logistic_times = read_txt("activation_function=logistic-time.txt")

    # 識別率をプロット
    accuracies_list = [tanh_accuracies, relu_accuracies, logistic_accuracies]
    plot_data(accuracies_list, "Accuracies")

    # 実行時間の記録をプロット
    times_list = [tanh_times, relu_times, logistic_times]
    plot_data(times_list, "Time [s]")

    # MNISTデータの読み込み
    (x, y), (xt, yt) = load_data()
    # 画像データの正規化
    x = data_reshape(x)
    xt = data_reshape(xt)

    # tanhのモデルによるヒートマップの生成
    heat_map('Results/retrain_data=60000-activation_function=tanh.sav', xt, yt)
