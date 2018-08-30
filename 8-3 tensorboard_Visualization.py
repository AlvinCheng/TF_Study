import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


# 載入手寫數字的數據庫 one_hot 把標籤轉乘0或1的格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定義運行次數
max_steps = 1001
# 圖片數量
image_num = 3000
# 文件路徑
DIR = 'D:/code/PycharmProjects/'

# 定義會話
sess = tf.Session()
# 載入圖片  stack 矩陣打包 將測試集的圖片0~3000 張打包起來存放在embedding
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

# 參數概要 tb
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)  # 計算var 平均值
        tf.summary.scalar('mean', mean)  # show 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  # 計算stddev 標準差 = (var - mean) ^ 2 再取平均
        tf.summary.scalar('stddev', stddev)  # show標準差
        tf.summary.scalar('max', tf.reduce_max(var))  # show最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # show最小值
        tf.summary.histogram('histogram', var)  # show直方圖


# 命名空間
with tf.name_scope('input'):
    # # 定義兩個placeholder
    x = tf.placeholder(tf.float32, [None, 784],
                       name='x-input')  # x 28 x 28 =784 一張的像數  拉長為784(列)  None = 傳入的批次 100(行) 的一維向量
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')  # y 是標籤 數字是0~9 所以標籤數 等於10


# 顯示圖片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1]) # 將 x 的形狀轉化成[-1, 28, 28, 1] -1 表任意值,784轉成28x28 維度1(黑白) 3就是(RGB)
    tf.summary.image('input', image_shaped_input, 10)

# 創建一個簡單的神經網路 輸入是784的神經元 輸出是標籤 10的神經元
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        w = tf.Variable(tf.zeros([784, 10]), name='w')
        variable_summaries(w)  # 傳入 w 到 summaries
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')  #
        variable_summaries(b)  # 傳入 b 到 summaries
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)  # 信號總和 經過 softmax 轉成為概略值 存放在 prediction

# 二次代價函數 線性曲線
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar('loss', loss)  # show loss

# 使用梯形下降法 學習率 0.2
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化變量
sess.run(tf.global_variables_initializer())

# 測試模型的準確率
# 使用 equal 將(y, 1) (prediction, 1) 比較結果 存放在一個correct_prediction布林行列表中 相同 為 true
# arg_max(y, 1) 求y標籤(真實樣本)裡面的最大的值是在那個位子,(prediction, 1) 求預測的標籤 最大值(概率)在那個位子
# 兩邊位子相同 返回 true
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # arg_max返回一維張亮中最大的值所在的位置
    # 求準確率
    with tf.name_scope('accuracy_mean'):
        # 使用 cast 將對比後結果轉換成32浮點型 true -> 1.0 false -> 0 再求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)  # show準確率

# 產生 metadata 文件 DeleteRecursively 遞迴砍目錄
# if tf.gfile.Exists(DIR+'projector/projector/metadata.tsv'):
#     tf.gfile.DeleteRecursively('D:/code/PycharmProjects/projector/projector')

with open(DIR+'projector/projector/metadata.tsv', 'w') as f:
    # arg_max求在一列元素中 那一個位置是最大值 test.labels 共有10個位置 經過 one_hot 只有0跟1
    # labels 格式 ex : 標籤是0 就是1000000000  標籤是1 就是0100000000
    lables = sess.run(tf.arg_max(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(lables[i]) + '\n')  # 隔一行 寫一個 lable


# 合併所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR+'projector\\projector\\', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR+'projector\\projector\\metadata.tsv'
embed.sprite.image_path = DIR+'projector\\data\\mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

#  開始訓練
# writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test", sess.graph)
for i in range(max_steps):
    batch_xs, batch_ys = mnist .train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options, run_metadata=run_metadata)  # 訓練時同時運行merged 統計
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
 # writer.add_summary(summary, i)  # 將 summary 跟運行週期 epoch 寫入文件

    # 每訓練100次 打印一次準確率
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(i) + "Testing Accuracy " + str(acc))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
