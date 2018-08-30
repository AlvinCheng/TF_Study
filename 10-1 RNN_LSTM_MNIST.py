import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入手寫數字的數據庫 one_hot 把標籤轉乘0或1的格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 輸入圖片是28*28
n_inputs = 28  # 輸入一行 一行有28個數據
max_time = 28  # 一共28行
lstm_size = 100  # 隱藏單元
n_classes = 10  # 10 個分類
# 每個批次大小 訓練模型時每次一次性的以矩陣模式放入50張
batch_size = 50  # 每個批次50個樣本
# 計算一共有多少個批次
n_batch = mnist.train.num_examples // batch_size  # num_examples 代表數據集訓練數據的數量有多少 整除(//)於 100 得到一共要訓練多少批次

# 定義兩個placeholder
# 這裡的None 表示第一個維度可以是任意的長度
x = tf.placeholder(tf.float32, [None, 784])  # x 28 x 28 =784 一張的像數  拉長為784(列)  None  = 傳入的批次 50(行) 的一維向量
# 正確的標籤
y = tf.placeholder(tf.float32, [None, 10])  # y 是標籤 數字是0~9 所以標籤數 等於10

# 初始化權值
# 使用 truncated_normal 截斷分布 方式初始化比較好 標準差 stddev 0.1
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 偏執值初始化
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定義 RNN 網路
def RNN(X, weights, biases):
    # inputs = [batch_size,max_time ,n_inputs] 轉換 格式 for dynamic_rnn 使用
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定義 LSTM 基本 CELL
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[state , batch_size , cell.state_size]
    # final_state[0] 是 cell state 是 cell block 中間的信號
    # final_state[1] 是 hidden state 是記錄最後一次 輸出的信號
    # outputs : The RNN output Tensor 記錄時間序列每一次的輸出結果 這案例有28個輸出 一圖片會向RNN傳入28次max_time
    #   If time_major == False (default), this will be a Tensor
    #      shaped: [batch_size, max_time, cell.out_size] ex: max_time =1 就是第1個序列的輸出結果 共有28個 第27次會等於 final_state[1]
    #   If time_major == True , this will be a Tensor
    #       shaped: [max_time,batch_size, cell.out_size]
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


# 解釋 outputs 跟 final_state 是多少維度 每個維度代表意義

# 計算RNN返回的結果 存放在 prediction
prediction = RNN(x, weights, biases)

# 損失函數 使用交叉商代價函數
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用AdamOptimizer 進行優化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 測試模型的準確率
# 使用 equal 將(y, 1) (prediction, 1) 比較結果 存放在一個correct_prediction布林行列表中 相同 為 true
# arg_max(y, 1) 求y標籤(真實樣本)裡面的最大的值是在那個位子,(prediction, 1) 求預測的標籤 最大值(概率)在那個位子
# 兩邊位子相同 返回 true
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max返回一維張亮中最大的值所在的位置
# 求準確率
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))  # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值
# 初始化變量
init = tf.global_variables_initializer()

# 開始訓練
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):  # 訓練 6 epoch 週期
        for batch in range(n_batch):  # 每一週期 就是 n_batch 共有多少批次  21 x n_batch
            # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得數據
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})  # 訓練數據

        acc = sess.run(accuracy,
                       feed_dict={x: mnist.test.images, y: mnist.test.labels})  # 執行 accuracy 來求準確率,傳入測試集的圖片跟標籤
        print("Iter" + str(epoch) + "Testing Accuracy " + str(acc))
