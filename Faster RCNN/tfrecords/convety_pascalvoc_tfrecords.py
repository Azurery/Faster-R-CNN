import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert2example(dataset_dir, name):
    filename = dataset_dir + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
            int(size.find('width').text),
            int(size.find('depth').text)]
    # Find annotations.
    # bboxes = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    labels_text = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        bbox = obj.find('bndbox')
        ymin.append(float(bbox.find('ymin').text) / shape[0])
        xmin.append(float(bbox.find('xmin').text) / shape[1])
        ymax.append(float(bbox.find('ymax').text) / shape[0])
        xmax.append(float(bbox.find('xmax').text) / shape[1])

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/class/label': int64_feature(labels),
            'image/object/class/label_text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)
            }))
    return example



def convert2tfrecords(original_dataset_dir, dataset_dir, year='2007', mode='trainval', samples_per_file=5000):
    # Dataset filenames, and shuffling.
    path = os.path.join(original_dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))

    i = 0
    index = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = '%s/pascalvoc_%s_%s_%02d.tfrecord' % (dataset_dir, year, mode, index)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < samples_per_file:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                image_name = filenames[i][:-4]
                example = _convert2example(original_dataset_dir, image_name)
                tfrecord_writer.write(example.SerializeToString())
                i += 1
                j += 1
            index += 1

    print('\nFinished converting the Pascal VOC dataset!')

if __name__ == '__main__':
    convert2tfrecords('/media/df416/84682e6b-3b02-4b6c-9746-82bcdb7a2ce9/软件/数据集/PASCAL VOC/VOCdevkit/VOC2007/',
                    'tfrecords/')