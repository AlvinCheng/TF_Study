import tensorflow as tf
import os
import tarfile
import requests

# inception 模型下載網址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 模型存放位置
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 獲取文件名 以及文件路徑
filename = inception_pretrain_model_url.split('/')[-1]  # 取"/"倒數第一個字串
filepath = os.path.join(inception_pretrain_model_dir, filename)

# >>> line = 'a+b+c+d'
# >>> line.split('+')
# ['a', 'b', 'c', 'd']
# >>> ['a', 'b', 'c', 'd'][-1]
# 'd'
# >>>

# 下載模型
if not os.path.exists(filepath):
    print("download:", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)

# 解壓縮文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# 模型結構存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb 是google 訓練好inception V3   的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')

with tf.Session() as sess:
    # 創建一個圖來存放google訓練好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # 保存圖的結構
    write = tf.summary.FileWriter(log_dir, sess.graph)
    write.close()

# tensorboard --logdir=D:\code\PycharmProjects\inception_log