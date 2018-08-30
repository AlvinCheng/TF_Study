import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入手寫數字的數據庫 one_hot 把標籤轉乘0或1的格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每個批次大小 訓練模型時每次一次性的以矩陣模式放入100張
batch_size = 100
# 計算一共有多少個批次
n_batch = mnist.train.num_examples // batch_size  # num_examples 代表數據集訓練數據的數量有多少 整除(//)於 100 得到一共要訓練多少批次


# 定義初始化權值形狀
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# conv2d 定義2維捲積層操作
def conv2d( x, W):
    # x 代表輸入是一個tensor(張量) 形狀為 [batch, in_height, in_height, in_channels] 的 4 維值
    # batch 代表 一個批次大小100 , in_height 代表圖片長, in_height 代表圖片寬, in_channels 1 :黑白 3 :彩色
    # W 代表是個濾波器(捲積核) 形狀是一個tensor [filter_height, filter_width, in_channels , out_channels]
    #  filter_height 代表濾波器長, filter_width 代表濾波器寬, in_channels 輸入通道數   out_channels 輸出通道數
    # strides 代表是步長 strides[0]跟strides[3] 規定填 1 ,strides[1]代表x方向的步長, strides[2]代表 y 方向的步長
    # padding 設置 SAME :採樣框超過原本平面超出補 0 "VALID" 不補0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max_pool_2x2 最大值池化層 2x2
def max_pool_2x2(x):
    # ksize [1, x, y, 1] 是窗口的大小 ksize[0]ksize[3] 規定填 1 ,ksize=[1, 2, 2, 1]是 2x2 的窗口
    # ksize [1, x, y, 1] 是窗口的大小 strides[0]strides[3] 規定填 1 ,strides=[1, 2, 2, 1]是 x,y 步長為 2 的 的池化掃描
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定義兩個placeholder
# x 28 x 28 =784 一張的像數  拉長為784(列)  None = 傳入的批次 100(行) 的一維向量
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])  # y 是標籤 數字是0~9 所以標籤數 等於10

# 改變 x 的格式轉為4D的向量[batch, in_height, in_width, in_channels] =>[-1, 28, 28, 1]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一個捲積層的權值跟偏置
# 5*5 的捲積採樣窗口,1 表示黑白 32:輸出是 使用32個捲積核從1個平面抽取特徵 採樣完會得到32個特徵平面
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  # 每一個捲積核一個偏置值

# 把 x_image 和權值向量進行conv2d捲積,再加上偏置值,然後應用於 relu 激活函數
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 進行max-pooling 池化

# 初始化第二個捲積層的權值跟偏置
W_conv2 = weight_variable([5, 5, 32, 64])  # 5*5 的採樣窗口,使用64個捲積核從32個平面抽取特徵 輸出會有64個特徵平面
b_conv2 = bias_variable([64])  # 每一個捲積核一個偏置值

# 把 h_conv1 和權值向量進行conv2d 捲積,再加上偏置值,然後應用於 relu 激活函數
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 進行max-pooling

# 28*28的圖片第一次捲積後還是28*28, 第一次池化後變成14*14
# 第2次捲積後為14*14, 第2次池化後變成7*7
# 經過上面操作後得到64張 7*7的平面

# 初始化第一個全連接層(兩個維度)的權值
w_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一層有7*7*64個神經元,全連接層的第一層有1024個神經元
b_fc1 = bias_variable([1024])  # 1024個節點

# 把池化層2的輸出扁平化為1維
# -1 表示任意值 在這邊表示100(一批次數量=100)
h_pool2_falt = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一個全連接層的輸出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_falt, w_fc1) + b_fc1)

# keep_prob 用來表示神經元的輸出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第2個全連階層 1024神經元 10代表有10個分類
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 計算輸出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 交叉商代價函數
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

# 開始訓練
with tf.Session() as sess:
    # 初始化變量
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):  # 訓練 21 epoch 週期
        for batch in range(n_batch):  # 每一週期 就是 n_batch 共有多少批次  21 x n_batch
            # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得數據
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})  # 訓練數據

        acc = sess.run(accuracy,
                       feed_dict={x: mnist.test.images, y: mnist.test.labels,
                                  keep_prob: 1.0})  # 執行 accuracy 來求準確率,傳入測試集的圖片跟標籤
        print("Iter" + str(epoch) + "Testing Accuracy " + str(acc))
