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


# 命名空間
with tf.name_scope("input"):
    # # 定義兩個placeholder
    x = tf.placeholder(tf.float32, [None, 784],
                       name='x-input')  # x 28 x 28 =784 一張的像數  拉長為784(列)  None = 傳入的批次 100(行) 的一維向量
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')  # y 是標籤 數字是0~9 所以標籤數 等於10

# 創建一個簡單的神經網路 輸入是784的神經元 輸出是標籤 10的神經元
with tf.name_scope("layer"):
    with tf.name_scope("wights"):
        w = tf.Variable(tf.zeros([784, 10]), name='w')
        variable_summaries(w)  # 傳入 w 到 summaries
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]), name='b')  #
        variable_summaries(b)  # 傳入 b 到 summaries
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x, w) + b
    with tf.name_scope("softmax"):
        prediction = tf.nn.softmax(wx_plus_b)  # 信號總和 經過 softmax 轉成為概略值 存放在 prediction

# 二次代價函數 線性曲線
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar('loss', loss)  # show loss

# 使用梯形下降法 學習率 0.2
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化變量
init = tf.global_variables_initializer()

# 測試模型的準確率
# 使用 equal 將(y, 1) (prediction, 1) 比較結果 存放在一個correct_prediction布林行列表中 相同 為 true
# arg_max(y, 1) 求y標籤(真實樣本)裡面的最大的值是在那個位子,(prediction, 1) 求預測的標籤 最大值(概率)在那個位子
# 兩邊位子相同 返回 true
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max返回一維張亮中最大的值所在的位置
    # 求準確率
    with tf.name_scope("accuracy_mean"):
        # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)  # show準確率

# 合併所有的summary
merged = tf.summary.merge_all()

#  開始訓練
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test", sess.graph)
    for epoch in range(51):  # 訓練 1 epoch 週期
        for batch in range(n_batch):  # 每一週期 就是 n_batch 共有多少批次  21 x n_batch
            # 使用rain.next_batch 每傳入 batch_size = 100 獲得一個批次的大小 batch_xs存放 數據 batch_ys 存放標籤(0-100,100-200)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 獲得數據
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})  # 訓練時同時運行merged 統計
            # merged 會有個返回值 存在 summary

        writer.add_summary(summary, epoch)  # 將 summary 跟運行週期 epoch 寫入文件
        # 執行 accuracy 來求準確率,傳入測試集的圖片跟標籤
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + "Testing Accuracy " + str(acc))

    # train_writer.close()

# 以上code可以優化 準確率提升到95%以上
# 可優化修改點
# 1.批次大小(100) 2.增加中間層 跟調整激活函數跟 神經元數量 3.修改 權值跟偏移值的初始值目前是zero初始化0
# 4.修改2次代價函數的方法 5.修改梯度學習率 跟替換其他演算 6.修改 訓練週期
