# 非線性回歸
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpyr 介於-0.5跟0.5之間生成200個隨機點 的一維數據 但是需要使用是2維所以在數據後面加一維度[:, np.newaxis] 使成為2維
# 隨機產生的數據會存在 [:, np.newaxis] 的 : 前面 得到 x_data 就是一各200行 一列的 數據
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]

# 生成干擾項隨機值 形狀是跟x_data一樣的
noise = np.random.normal(0, 0.02, x_data.shape)

# y_data = x_data 平方 + 噪音隨機值
y_data = np.square(x_data) + noise
#
# 定義兩個placeholder (佔位符號)
x = tf.placeholder(tf.float32, [None, 1])  # float32 是類型 後面None是任意形狀[] 行是不確定 但是 列是一列 對應上面的x_data
y = tf.placeholder(tf.float32, [None, 1])  # 案例 x_data 傳入200行 None = 200

# 建構神經網路 考慮輸入一個x 經過一個神經網路計算  最後會得到一個y 這是y預測值 希望這y 會貼近真實值的y 接近
# 輸入層是一個點就是一個神經元 中間層是可以調整的 案例使用10個神經元做園中間層 輸出層也是一個神經元

# 拓建一個神經網絡中間層
Weigths_L1 = tf.Variable(tf.random_normal([1, 10]))  # 定義權值L1是tf的隨機變量 形狀是一行10列 1代表輸入 10代表中間層是10個神經元所以是10
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 定義偏移植是tf的隨機變量 形狀是一行10列 1代表輸入 10代表中間層是10個神經元所以是10
Wx_plus_b_L1 = tf.matmul(x, Weigths_L1) + biases_L1  # x 矩陣 乘於 權值(Weigths_L1)矩陣 + biases_L1 等於信號總和
L1 = tf.nn.tanh(Wx_plus_b_L1) # 使用 雙曲正切函數來作為激活函數 作用於信號的總和 得到L1 是 中間層的輸出

# 定義神經網路輸出層
Weigths_L2 = tf.Variable(tf.random_normal([10, 1]))  # 定義權值L2是一個10行一列的 (因為中間層10的神經元 輸出層一個)
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 輸出是一個神經元所以是一個 bias

# 將中間層的輸出(就等於是輸出層的輸入 L1) 矩陣相乘輸出層的權值 Weigths_L2 _+上 biases_L2 得到 信號總和
Wx_plus_b_L2 = tf.matmul(L1, Weigths_L2) + biases_L2 # 優化時會改變 Weigths_L2 biases_L2 Weigths_L1 biases_L1 這4個變量

prediction = tf.nn.tanh(Wx_plus_b_L2)  # 用雙曲正切函數來作為激活函數 得到 預測結果 prediction

# 二次代價函數
loss =  tf.reduce_mean(tf.square(y-prediction))

# 使用梯度下降法訓練 學習率0.1 最小化 loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 定義會話
with tf.Session() as sess:
    # 變量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):  # 訓練2000次
        # 傳回值 只用feed 方式 賦值 x 使用x_data傳入 y 使用 y_data 傳入 ,(y_data ,x_data 是樣本底)
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 獲得預測值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    train_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test", sess.graph)
    train_writer.close()

    # 繪圖
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

