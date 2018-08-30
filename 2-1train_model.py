import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 產生100 隨機點
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2   # 真實值 0.1是斜率 0.2是級距


# 構造一個線性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b  # (預測值)

# 二次代價函數 loss 越小代表(預測值)越接近(真實值)
loss = tf.reduce_mean(tf.square(y_data - y))
# 定義一個梯度下降法來進行訓練的優化器 訓練 k 要越接近 0.1是斜率,  b要接近 0.2是級距
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代價函數
train = optimizer.minimize(loss)
# 初始化變量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 訓練201次
    plt.figure()
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:  # 每20打印一次結果
             print(step, sess.run([k, b]))

    train_writer = tf.summary.FileWriter("D:/code/PycharmProjects/tfboard_Test", sess.graph)
    train_writer.close()

    # plt.scatter(x_data, y_data)
    # plt.scatter(x_data, y_data)
    # plt.show()
