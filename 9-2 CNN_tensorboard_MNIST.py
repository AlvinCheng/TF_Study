import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 載入手寫數字的數據庫 one_hot 把標籤轉乘0或1的格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每個批次大小 訓練模型時每次一次性的以矩陣模式放入100張
batch_size = 100
# 計算一共有多少個批次
n_batch = mnist.train.num_examples // batch_size  # num_examples 代表數據集訓練數據的數量有多少 整除(//)於 100 得到一共要訓練多少批次


# 參數概要 tb
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)  # 計算var 平均值
        tf.summary.scalar('mean', mean)  # show 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  # 計算stddev 標準差 = (var - mean) ^ 2 再取平均
        tf.summary.scalar('stddev', stddev)  # show標準差
        tf.summary.scalar('max', tf.reduce_max(var))  # show最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # show最小值
        tf.summary.histogram('histogram', var)  # show直方圖


# 定義初始化權值形狀
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 初始化偏置值
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# conv2d 定義2維捲積層操作
def conv2d(x, W):
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


# 命名空間
with tf.name_scope("input"):
    # 定義兩個placeholder
    # x 28 x 28 =784 一張的像數  拉長為784(列)  None = 傳入的批次 100(行) 的一維向量
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')  # y 是標籤 數字是0~9 所以標籤數 等於10
    with tf.name_scope("x_image"):
        # 改變 x 的格式轉為4D的向量[batch, in_height, in_width, in_channels] =>[-1, 28, 28, 1]
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope("Conv1"):
    # 初始化第一個捲積層的權值跟偏置
    # 5*5 的捲積採樣窗口,1 表示黑白 32:輸出是 使用32個捲積核從1個平面抽取特徵 採樣完會得到32個特徵平面
    with tf.name_scope("w_conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32], name='w_conv1')
    with tf.name_scope("b_conv1"):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一個捲積核一個偏置值

    with tf.name_scope("conv2d_1"):
        # 把 x_image 和權值向量進行conv2d捲積,再加上偏置值,然後應用於 relu 激活函數
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope("h_pool1"):
        h_pool1 = max_pool_2x2(h_conv1)  # 進行max-pooling 池化

with tf.name_scope("Conv2"):
    with tf.name_scope("W_conv2"):
        # 初始化第二個捲積層的權值跟偏置
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5 的採樣窗口,使用64個捲積核從32個平面抽取特徵 輸出會有64個特徵平面
        with tf.name_scope("b_conv2"):
            b_conv2 = bias_variable([64], name='b_conv2')  # 每一個捲積核一個偏置值

    with tf.name_scope("conv2d_2"):
        # 把 h_conv1 和權值向量進行conv2d 捲積,再加上偏置值,然後應用於 relu 激活函數
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope("h_conv2"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("h_pool2"):
        h_pool2 = max_pool_2x2(h_conv2)  # 進行max-pooling

# 28*28的圖片第一次捲積後還是28*28, 第一次池化後變成14*14
# 第2次捲積後為14*14, 第2次池化後變成7*7
# 經過上面操作後得到64張 7*7的平面


with tf.name_scope("fc1"):
    with tf.name_scope("w_fc1"):
        # 初始化第一個全連接層(兩個維度)的權值
        w_fc1 = weight_variable([7 * 7 * 64, 512], name='w_fc1')  # 上一層有7*7*64個神經元,全連接層的第一層有1024個神經元
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([512], name='b_fc1')  # 1024個節點

    with tf.name_scope("h_pool2_falt"):
        # 把池化層2的輸出扁平化為1維
        # -1 表示任意值 在這邊表示100(一批次數量=100)
        h_pool2_falt = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_falt')
        # 求第一個全連接層的輸出
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_falt, w_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    with tf.name_scope("keep_prob"):
        # keep_prob 用來表示神經元的輸出概率
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope("h_fc1_dorp"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_dorp')

with tf.name_scope("fc2"):
    with tf.name_scope("w_fc2"):
        # 初始化第2個全連階層 1024神經元 10代表有10個分類
        w_fc2 = weight_variable([512, 10], name='w_fc2')
        with tf.name_scope("b_fc2"):
            b_fc2 = bias_variable([10], name='b_fc2')

    with tf.name_scope("prediction"):
        with tf.name_scope("wx_plus_b2"):
            wx_plus_b2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        with tf.name_scope("softmax"):
            prediction = tf.nn.softmax(wx_plus_b2)  # 計算輸出

# 交叉商代價函數
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope("train"):
    # 使用AdamOptimizer 進行優化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 測試模型的準確率
# 使用 equal 將(y, 1) (prediction, 1) 比較結果 存放在一個correct_prediction布林行列表中 相同 為 true
# arg_max(y, 1) 求y標籤(真實樣本)裡面的最大的值是在那個位子,(prediction, 1) 求預測的標籤 最大值(概率)在那個位子
# 兩邊位子相同 返回 true
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max返回一維張亮中最大的值所在的位置

    # 求準確率
    with tf.name_scope("accuracy__mean"):
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))  # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值
        tf.summary.scalar('accuracy', accuracy)

# 合併所有的summary
merged = tf.summary.merge_all()

# 開始訓練
with tf.Session() as sess:
    # 初始化變量
    sess.run(tf.global_variables_initializer())
    # train_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test/", sess.graph)
    # test_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test/", sess.graph)
    train_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test/train", sess.graph)
    test_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test/test", sess.graph)
    for i in range(1001):
        # 訓練模型
        # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得訓練集數據
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})  # 訓練數據
        # 訓練時同時運行merged 紀錄訓練集(train)計算的參數 統計 merged 會有個返回值 存放在 summary
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)  # 將 summary 跟運行週期 epoch 寫入文件

        batch_xs, batch_ys = mnist.test.next_batch(batch_size)  # 獲得測試集數據
        # 訓練時同時運行merged 紀錄測試集(test)計算的參數 統計 merged 會有個返回值 存放在 summary
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)  # 將 summary 跟運行週期 epoch 寫入文件

        # 每訓練100次 打印一次準確率
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print("Iter" + str(i) + "Testing Accuracy " + str(test_acc) + ", Training Accuracy" + str(train_acc))
