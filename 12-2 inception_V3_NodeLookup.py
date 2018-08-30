# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

# 模型存放位置
model_dir = "inception_model"


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path ='inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path: object) -> object:
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
        # 加載分類字符串n********對應分類名稱的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 一行一行讀取數據
        #  EX: n01491361    tiger shark, Galeocerdo cuvieri
        # n01491361
        # tiger shark, Galeocerdo cuvieri
        for line in proto_as_ascii_lines:  # 一行一行 讀入 比對
            # 去掉換行符
            line = line.strip('\n')
            # 按照'\t'分割 "tab" 鍵
            parsed_items = line.split('\t')
            # 獲取分類編號
            uid = parsed_items[0]
            # 獲取分類名稱
            human_string = parsed_items[1]
            # 保存編號字符串n ********與分類名稱映射關係
            uid_to_human[uid] = human_string

        # 加載分類字符串n ********對應分類編號1-1000的文件
        # EX:
        # {
        #     target_class: 443
        #     target_class_string: "n01491361"
        # }
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                # 獲取分類編號1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                # 獲取編號字符串n********
                target_class_string = line.split(': ')[1]
                # 保存分類編號1-1000與編號字符串n********映射關係
                node_id_to_uid[target_class] = target_class_string[1:-2]  # 去掉首尾的双引号  ex :"n01491361"


        # 建立分類編號1-1000對應分類名稱的映射關係
        # 建立node_id_to_name 鍵值對字典 映射關係為[key,val]=[443, "tiger shark, Galeocerdo cuvieri"]
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 獲取分類名稱
            name = uid_to_human[val]  # 把 n01491361 值帶入到 uid_to_human 得到  name
            # 利用分類名稱name來重新建立一個新的字典 : 建立分類編號1-1000到分類名稱的映射關係
            node_id_to_name[key] = name
        return node_id_to_name

    # 傳入分類編號1-1000返回分類名稱
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 創建一個圖來存放google訓練好的模型 ==> Inception-v3模型
with tf.gfile.FastGFile(os.path.join(model_dir,
                                     'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='') # 將 V3 import 入當前 graph

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # V3 graph 的最後一個輸出名稱是softmax [1, 1008]
    # 遍歷目錄
    # root=images/' dirs =子目錄  files= 裡面的圖片
    for root, dirs, files in os.walk('images/'):
        for file in files:  # 循環裡面的圖片
            #  載入圖片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()

            #  傳入image_data圖片 進行計算  DecodeJpeg 圖片格式是jpg格式 將計算結果(2維)存放在 predictions
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 圖片格式是jpg格式
            # predictions = sess.run(softmax_tensor,{'DecodeGif/contents:0': image_data})#圖片格式是gif格式
            print(predictions.shape)  # [1, 1008]
            predictions = np.squeeze(predictions)  # 把結果轉為1維數據
            print(predictions.shape)  # [1008 ,]
            # 打印圖片路徑及名稱
            image_path = os.path.join(root, file)
            print("image_path = " + image_path)
            # 顯示圖片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 把預設結果做排序 因為有1000個結果不可能都打印所以要排序
            top_k = predictions.argsort()[-5:][::-1]  # argsort 是從小到大[-5:]取概率最後5名 ,[::-1] 修改為從大到小再存入 top_k
            print("NodeLookup start ")
            print(top_k)
            print("top 5 id num=" + str(top_k))
            node_lookup = NodeLookup()
            # node_id 就是從1到1000的一個編號 top_k 從大到小的5個結果
            for node_id in top_k:
                print('alvin1', node_id)
                # 獲取分類名稱
                human_string = node_lookup.id_to_string(node_id)
                # 獲取該分類的置信度 分類的百分比
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
