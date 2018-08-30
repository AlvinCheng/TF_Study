import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 載入手寫數字的數據庫 one_hot 把標籤轉乘0或1的格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每個批次大小 訓練模型時每次一次性的以矩陣模式放入100張
batch_size = 100
# 計算一共有多少個批次
n_batch = mnist.train.num_examples // batch_size  # num_examples 代表數據集訓練數據的數量有多少 整除(//)於 100 得到一共要訓練多少批次

# 定義兩個placeholder
x = tf.placeholder(tf.float32, [None, 784])  # x 28 x 28 =784 一張的像數  拉長為784(列)  None = 傳入的批次 100(行) 的一維向量
y = tf.placeholder(tf.float32, [None, 10])   # y 是標籤 數字是0~9 所以標籤數 等於10
keep_prob = tf.placeholder(tf.float32)  # 定義 Dropout 參數

# 創建一個簡單的神經網路 輸入是784的神經元 輸出是標籤 10的神經元
# w = tf.Variable(tf.zeros([784, 10]))  # 使用 zeros 這種初始化方式不好
# b = tf.Variable(tf.zeros([10]))  #

# 故意定義2000個神經元 跟數個中間層 來驗證Dropout :太複雜的神經網路 來跑太少的樣本 會出現 過擬合(Dropout)
w1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))  # 使用 truncated_normal 截斷分布 方式初始化比較好 標準差 stddev 0.1
b1 = tf.Variable(tf.zeros([2000])+0.1)  # 偏執值初始化 zeros+0.1 = 0+0.1 =0.1 意思
L1 = tf.nn.tanh(tf.matmul(x, w1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

# 故意定義2000個神經元 跟數個中間層 來驗證Dropout :太複雜的神經網路 來跑太少的樣本 會出現 過擬合(Dropout)
w2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))  # 使用 truncated_normal 截斷分布 方式初始化比較好 標準差 stddev 0.1
b2 = tf.Variable(tf.zeros([2000])+0.1)  # 偏執值初始化 zeros+0.1 = 0+0.1 =0.1 意思
L2 = tf.nn.tanh(tf.matmul(L1_drop, w2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 故意定義1000個神經元 跟數個中間層 來驗證Dropout :太複雜的神經網路 來跑太少的樣本 會出現 過擬合(Dropout)
w3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))  # 使用 truncated_normal 截斷分布 方式初始化比較好 標準差 stddev 0.1
b3 = tf.Variable(tf.zeros([1000])+0.1)  # 偏執值初始化 zeros+0.1 = 0+0.1 =0.1 意思
L3 = tf.nn.tanh(tf.matmul(L2_drop, w3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

# 輸出層 10個神經元  784x2000x2000x1000x10
w4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))  # 使用 truncated_normal 截斷分布 方式初始化比較好 標準差 stddev 0.1
b4 = tf.Variable(tf.zeros([10])+0.1)  # 偏執值初始化 zeros+0.1 = 0+0.1 =0.1 意思
prediction = tf.nn.softmax(tf.matmul(L3_drop, w4) + b4)  # 信號總和 經過 softmax 轉成為概略值 存放在 prediction

# 二次代價函數 線性曲線
# loss = tf.reduce_mean(tf.square(y - prediction))
# 使用交叉墒 作為代價函數 時機=>使用softmax 或 激活函數 sigmoid 或是 S 曲線
# labels 標籤 是y logits 是預測值 prediction
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯形下降法 學習率 0.2
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 測試模型的準確率
# 使用 equal 將(y, 1) (prediction, 1) 比較結果 存放在一個correct_prediction布林行列表中 相同 為 true
# arg_max(y, 1) 求y標籤(真實樣本)裡面的最大的值是在那個位子,(prediction, 1) 求預測的標籤 最大值(概率)在那個位子
# 兩邊位子相同 返回 true
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max返回一維張亮中最大的值所在的位置
# 求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值

# 開始訓練
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31): # 訓練 31 epoch 週期
        for batch in range(n_batch): # 每一週期 就是 n_batch 共有多少批次  21 x n_batch
            # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得數據
            sess.run(train_step, feed_dict={x:batch_xs, y: batch_ys, keep_prob: 0.7}) # 訓練keep_prob=1.0=100%神經元0.7=70%

        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0}) # 測試使用1.0
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0}) #

        print("Iter" + str(epoch) + "Testing Accuracy " + str(test_acc) + "Training Accuracy" + str(train_acc))




# 以上code可以優化 準確率提升到95%以上
# 可優化修改點
# 1.批次大小(100) 2.增加中間層 跟調整激活函數跟 神經元數量 3.修改 權值跟偏移值的初始值目前是zero初始化0
# 4.修改2次代價函數的方法 5.修改梯度學習率 跟替換其他演算 6.修改 訓練週期
