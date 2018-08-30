import tensorflow as tf
import os
import random
import math
import sys

#################### 將原始 數據圖片 轉成 TFRecord 格式 #####################

# 驗證集數量
_NUM_TEST = 500
# 隨機種子
_RANDOM_SEED = 0
# 數據模塊
_NUM_SHARDS = 5
# 數據集路徑
DATASET_DIR = "D:/code/PycharmProjects/slim/images"
# 標籤文件名子
LABLES_FILENAME = "D:/code/PycharmProjects/slim/images/lables.txt"


# 定義 tfrecord 文件的路徑+名子  做字符串的組合處理  split_name : 訓練或是測試  shard_id:分配數據塊 ID 值從0到4  _NUM_SHARDS:數據塊 總數
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filemane = 'image_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filemane)


# 判斷tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            # 定義 tfrecord 文件的路徑+名子
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

        if not tf.gfile.Exists(output_filename):
            return False
    return True


# 獲取所有文件以及分類
def _get_filenames_and_classes(dataset_dir):
    # 數據目錄
    directories = []
    # 分類名稱
    class_names = []
    for filename in os.listdir(dataset_dir):  # 在dataset_dir 路徑下 尋找 filename 文件
        # 合併文件路徑
        path = os.path.join(dataset_dir, filename)
        # 判斷該路徑是否為目錄
        if os.path.isdir(path):
            # 加入數據目錄
            directories.append(path)  # 把找到文件夾下面的目錄跟檔案名稱保存下來 directories class_names 分類名稱
            # 加入類別名稱
            class_names.append(filename)

    photo_filenames = []
    # 循環每個分類的文件夾
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            # 把圖片加入圖片列表
            photo_filenames.append(path)

    return photo_filenames, class_names


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, class_id):
    # Abstuact base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={  # 固定寫法
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
    }))


def write_label_file(labels_to_class_names, dataset_dir, filename=LABLES_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


# 把數據轉為TFRecord 格式
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    # 計算每一個數據塊有多少數據
    num_per_shard = int(len(filenames) / _NUM_SHARDS)  # 若數據量小 不用切分成數據塊
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                # 定義 TFRecord 文件的路徑+名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    # 每一個數據塊開始的位置
                    start_ndx = shard_id * num_per_shard
                    # 每一個數據塊最後的位置
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):  # 循環每個數據塊 開始到最後的位置
                        try:
                            sys.stdout.write('\r>> Converting image %d%d shard %d' % (i + 1, len(filenames), shard_id))
                            sys.stdout.flush()
                            # 讀取圖片
                            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                            # 獲得圖片的類別名稱
                            class_name = os.path.basename(os.path.dirname(filenames[i]))  # class_name 圖片分類
                            # 找到類別名稱對應的id
                            class_id = class_names_to_ids[class_name]  # class_id 圖片id
                            # 生成 tfrecord 文件
                            example = image_to_tfexample(image_data, b'jpg',
                                                         class_id)  # image_data :圖片數據  b'jpg' : 圖片格式  class_id: 圖片ID
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read:", filenames[i])
                            print("Error:", e)
                            print("Skip it \n")

    # sys.stdout_write('\n')
    # sys.stdout.flush()


if __name__ == '__main__':
    # 判斷tfrecord 文件是否存在
    if _dataset_exists(DATASET_DIR):
        print('TFRecord File exists')
    else:
        # 獲得所有圖片以及分類
        photo_filename, class_names = _get_filenames_and_classes(DATASET_DIR)
        # 把分類轉為字典格式, 類似於{'catdog': 3, 'flower': 1, 'potholed': 4, 'waffled': 2, 'gauzy':, 0}
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        # 把數據切分為訓練集和測試集
        random.seed(_RANDOM_SEED)  #
        random.shuffle(photo_filename)  # shuffle 把 數據打亂
        training_filenames = photo_filename[_NUM_TEST:]  # 前0~500
        testing_filenames = photo_filename[:_NUM_TEST]  # 500~1000 看幾張圖片

        # 數據轉換
        _convert_dataset('train', training_filenames, class_names_to_ids, DATASET_DIR)
        _convert_dataset('test', testing_filenames, class_names_to_ids, DATASET_DIR)

        # 輸出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, DATASET_DIR)
