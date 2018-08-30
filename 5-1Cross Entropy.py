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

# 創建一個簡單的神經網路 輸入是784的神經元 輸出是標籤 10的神經元
w = tf.Variable(tf.zeros([784, 10]))  #
b = tf.Variable(tf.zeros([10]))  #
prediction = tf.nn.softmax(tf.matmul(x, w) + b)  # 信號總和 經過 softmax 轉成為概略值 存放在 prediction

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
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1)) # arg_max返回一維張亮中最大的值所在的位置
# 求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值

# 開始訓練
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): # 訓練 21 epoch 週期
        for batch in range(n_batch): # 每一週期 就是 n_batch 共有多少批次  21 x n_batch
            # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得數據
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})  # 訓練數據

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}) # 執行 accuracy 來求準確率,傳入測試集的圖片跟標籤
        print("Iter" + str(epoch) + "Testing Accuracy " + str(acc))


# 以上code可以優化 準確率提升到95%以上
# 可優化修改點
# 1.批次大小(100) 2.增加中間層 跟調整激活函數跟 神經元數量 3.修改 權值跟偏移值的初始值目前是zero初始化0
# 4.修改2次代價函數的方法 5.修改梯度學習率 跟替換其他演算 6.修改 訓練週期
