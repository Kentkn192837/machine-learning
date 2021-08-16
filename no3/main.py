import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def most_likely_index(pre_list):
    answer = np.empty(len(pre_list), dtype='int32')
    for i, val in enumerate(pre_list):
        answer[i] = np.argmax(val)
    return answer

def img_reshape(img_list):
    new_image = np.empty((len(img_list), 64, 64), dtype='uint8')
    for i, _ in enumerate(img_list):
        new_image[i] = cv2.resize(img_list[i], dsize=(64, 64))
    return new_image

def data_reshape(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

def plot_acc( ac, val_ac ):
    plt.plot( ac )
    plt.plot( val_ac )
    plt.title('epochs-accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()

def plot_loss( loss, val_loss ):
    plt.plot( loss )
    plt.plot( val_loss )
    plt.title('epochs-loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['loss', 'validation_loss'])
    plt.show()

def heat_map(matrix):
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix, annot=True, fmt='2d', cmap='Reds')
    plt.show()

def create_model(class_num):
    input_tensor = Input( shape=(64, 64, 1) )
    vgg = VGG16( include_top=False, input_tensor=input_tensor, weights=None )
    x = vgg.output
    x = Flatten()(x)
    x = Dense(class_num, activation="softmax")(x)
    return Model( inputs=vgg.inputs, outputs=x )

if __name__ == '__main__':
    # parameters--------------------
    random_state = 60  # 乱数生成時の種値
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_num = len(class_names) # 分類するクラスの数
    BATCH_SIZE = 4  # バッチサイズの値
    EPOCH = 200  # 学習するepoch数
    lr = 1e-4  # 最適化アルゴリズムAdamに与える学習率
    #-------------------------------

    # Fashion-MNISTデータを取得
    (x_train, y_train), (x_test, y_test) = load_data()

    # 訓練データを6000件確保
    x_train, _, y_train, _ = train_test_split(x_train, y_train,
                                              test_size=0.9,
                                              random_state=random_state
                                              )
    # テストデータを1000件確保
    vertification_train, x_test, vertification_test, y_test = train_test_split(x_test, y_test,
                                                                               test_size=0.1,
                                                                               random_state=random_state
                                                                               )
    # テストデータとして利用しなかったデータのうち100件を検証用データとして確保
    _, vertification_train, _, vertification_test = train_test_split(vertification_train,
                                                                     vertification_test,
                                                                     test_size=100,
                                                                     random_state=random_state
                                                                    )
    
    # (28,28)ではVGG16で学習できないので、(64,64)に変換
    x_train = img_reshape(x_train)
    x_test = img_reshape(x_test)
    vertification_train = img_reshape(vertification_train)

    # データを正規化
    x_train = data_reshape(x_train)
    x_test = data_reshape(x_test)
    vertification_train = data_reshape(vertification_train)

    # one-hot表現に変換
    y_train = to_categorical( y_train, class_num )
    y_test = to_categorical( y_test, class_num )
    vertification_test = to_categorical( vertification_test, class_num )

    # モデル生成
    model = create_model(class_num)

    # モデルの構造を出力
    model.summary()

    model.compile(optimizer=Adam( learning_rate=lr ),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 一定のepochの間バリデーションロスが下がらなかった場合、学習を中断する
    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0.001,
                                  patience=10, verbose=1)
    
    # 学習を実行
    hist = model.fit( x=x_train, y=y_train, batch_size=BATCH_SIZE,
                      epochs=EPOCH, verbose=1, validation_data=(x_test, y_test),
                      callbacks=[earlystopping])
    
    # 学習による正解率,損失関数の値の推移のグラフを出力
    plot_acc( hist.history["accuracy"], hist.history["val_accuracy"] )
    plot_loss( hist.history["loss"], hist.history["val_loss"] )

    # モデルによる予測データを取得
    predict = model.predict(vertification_train)

    # モデルの予測で最も可能性が高いクラスのインデックス番号のリストを取得
    predict_index = most_likely_index(predict)
    true_class = most_likely_index(vertification_test)

    # 混同行列のヒートマップを出力
    matrix = confusion_matrix(predict_index, true_class)
    heat_map(matrix)

    # 適合率, 再現率, F値, 正解率, マクロ平均, マイクロ平均を出力
    print(classification_report(true_class, predict_index))
    