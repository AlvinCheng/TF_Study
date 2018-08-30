import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

lines = tf.gfile.GFile('retrain/output_labels.txt').readlines()
uid_to_human = {}
# 一行一行讀取數據
for uid, line in enumerate(lines):
    # 去掉換行符
    line = line.strip('\n')
    uid_to_human[uid] = line
    print(uid_to_human[uid])


def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


# 模型結構存放文件
log_dir = 'retrain/retrain_inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# with tf.gfile.FastGFile('retrain/output_graph.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['Placeholder'], ['final_result'],
#                                                                   tf.float32.as_datatype_enum)
#     graph_def = TransformGraph(graph_def, ['module_apply_default/hub_input/Sub'], ['final_result'],
#                                ['remove_nodes(op=PlaceholderWithDefault)',
#                                 'strip_unused_nodes(type=float, shape=\"1,224,224,3\")', 'sort_by_execution_order'])
#     with tf.gfile.FastGFile('retrain/output_graph.pb', 'wb') as f:
#         f.write(graph_def.SerializeToString())

with tf.Session() as sess:
    # 創建一個圖來存放 我們 retrain 後的模型 pb  output_graph.pb
    with tf.gfile.FastGFile('retrain/output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        #  保存圖的結構
        write = tf.summary.FileWriter(log_dir, sess.graph)
        write.close()

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # 遍歷目錄
    for root, dirs, files in os.walk('retrain/input_images_test'):
        for file in files:
            # 載入圖片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 圖片格式是jpg格式
            # data_tensor = tf.get_default_graph().get_tensor_by_name('DecodeJpeg/contents:0')
            # predictions = sess.run(softmax_tensor, {data_tensor: image_data})
            print(predictions.shape)
            predictions = np.squeeze(predictions)  # 把結果轉為1維數據
            print(predictions.shape)
            # 打印圖片路徑以及名稱
            image_path = os.path.join(root, file)
            print(image_data)
            # 顯示圖片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序
            top_k = predictions.argsort()[::-1]
            print("top id num=" + str(top_k))
            for node_id in top_k:
                # 獲取分類名稱
                human_string = id_to_string(node_id)
                # 獲取該分類置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()


