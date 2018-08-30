import tensorflow as tf

A = tf.constant(50, name='const_A')
B = tf.constant(100, name='const_B')

with tf.name_scope('Add'):
    C = A+B

with tf.Session() as sess:
    with tf.name_scope('Run'):
        D = sess.run(C*3)
    print(D)

    train_writer = tf.summary.FileWriter('D:/code/PycharmProjects/tfboard_Test', sess.graph)
    train_writer.close()

