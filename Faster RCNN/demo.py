import tensorflow as tf
import numpy as np
import tqdm
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


def parse_tfreocrd(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                    features={'image/height': tf.FixedLenFeature([1], tf.int64),
                                                'image/width': tf.FixedLenFeature([1], tf.int64),
                                                'image/encoded':  tf.FixedLenFeature([1], tf.string),
                                                'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                                                'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                                                'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                                                'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                                                'image/object/class/label': tf.VarLenFeature(tf.int64),
                                                'image/object/class/text': tf.VarLenFeature(tf.string)})

    xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'])
    # print(xmin)
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    label = tf.sparse_tensor_to_dense(features['image/object/class/label'])
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    height = features['image/height'][0]
    width = features['image/width'][0]
    bboxes = tf.transpose(tf.stack((xmin, xmax, ymin, ymax)), name='bboxes')
    return image, bboxes, label

if __name__ == '__main__':
    tf_record_list = ['tfrecords/pascalvoc_2007_trainval_00.tfrecord', 'tfrecords/pascalvoc_2007_trainval_01.tfrecord']
    dataset = tf.data.TFRecordDataset('tfrecords/pascalvoc_2007_trainval_01.tfrecord').map(parse_tfreocrd)
    dataset = dataset.batch(1).repeat(1)
    #print(dataset)
    # iterator = dataset.make_one_shot_iterator()
    # data = iterator.get_next()
    # print(data[4])
    for image, bboxes, labels in tfe.Iterator(dataset):
        print(labels)
    