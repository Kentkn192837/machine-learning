import numpy as np
import time
import pickle
from tensorflow.keras.datasets.mnist import load_data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def data_reshape(x):
    x = (x.reshape(len(x), 28 * 28)).astype(np.float64)
    x /= x.max()
    return x

def model_save(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def save_result(result_list, filename):
    with open(filename, mode='w') as f:
        for d in result_list:
            f.write("%s\n" % d)

if __name__ == '__main__':
    # MNISTデータの読み込み
    (x, y), (xt, yt) = load_data()

    # 画像データの正規化
    x = data_reshape(x)
    xt = data_reshape(xt)

    # 何%を検証用にするか
    nVals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # train_test_split関数に与える乱数の種値
    random_state = 60

    # 利用する活性化関数
    activation_function_list = ['tanh', 'relu', 'logistic']

    for activation_function in activation_function_list:
        accuracies = []  # 正確度保存用
        proc_time = []   # 実行時間保存用
        for per in nVals:

            if per == 1:
                trainData = x
                trainLabels = y
            else:
                trainData, valData, trainLabels, valLabels = train_test_split(x, y,
                                                                        test_size=1-per,
                                                                        random_state=random_state
                                                                    )

            start = time.time()  # 時間計測開始
            model = MLPClassifier(hidden_layer_sizes=(100,),
                                activation=activation_function,
                                max_iter=1000
                                )
            model.fit(trainData, trainLabels)
            score = model.score(xt, yt)
            duration = time.time() - start  # 計測終了

            accuracies.append(score)
            proc_time.append(duration)

            # モデルを保存
            filename = 'train_data='+ str(len(trainData)) + '-activation_function=' + activation_function + '.sav'
            model_save(model, filename)

        # 識別の結果を保存
        filename = 'activation_function=' + activation_function + '-accuracy.txt'
        save_result(accuracies, filename)

        # 実行時間を保存
        filename = 'activation_function=' + activation_function + '-time.txt'
        save_result(proc_time, filename)
