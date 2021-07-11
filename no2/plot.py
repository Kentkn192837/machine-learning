import pickle
import numpy as np
from matplotlib import pyplot as plt

def read_txt(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = float(lines[i].rstrip('\n'))
    return lines

def model_load():
    return pickle.load(open(filename, 'rb'))

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
