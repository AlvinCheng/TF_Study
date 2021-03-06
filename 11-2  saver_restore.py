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
y = tf.placeholder(tf.float32, [None, 10])  # y 是標籤 數字是0~9 所以標籤數 等於10

# 創建一個簡單的神經網路 輸入是784的神經元 輸出是標籤 10的神經元
w = tf.Variable(tf.zeros([784, 10]))  #
b = tf.Variable(tf.zeros([10]))  #
prediction = tf.nn.softmax(tf.matmul(x, w) + b)  # 信號總和 經過 softmax 轉成為概略值 存放在 prediction

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
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))  # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值

# 使用 Saver 模型套件
saver = tf.train.Saver()

# 開始訓練
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # 載入模型
    saver.restore(sess, 'net/my_net.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
