import os
import numpy as np
import tensorflow as tf
import cv2
from functools import partial
from preprocessing.preprocessing import image_argument_with_imgaug, preprocessing_training_func

# training
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
    xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'])
    ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'])
    ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'])
    label = tf.sparse_tensor_to_dense(features['image/object/class/label'])
    image = tf.image.decode_jpeg(features['image/encoded'][0])
    height = features['image/height'][0]
    width = features['image/width'][0]
    bboxes = tf.transpose(tf.stack((xmin, xmax, ymin, ymax)), name='bboxes')
    return image, bboxes, height, width, label
    
def get_dataset(tf_record_list, 
                batch_size=1,
                repeat_size=1,
                shuffle_size=1000,
                shuffle=True,
                prefetch=False,
                prefecth_size=1000,
                min_size=600,
                max_size=1000,
                preprocessing_type='caffe',
                caffe_pixel_means=None,
                data_argumentation=True,
                iaa_sequence=None
                ):
    dataset = tf.data.TFRecordDataset(tf_record_list).map(parse_tfreocrd)
    
    if data_argumentation:
        image_argument_partial = partial(image_argument_with_imgaug, iaa_sequence=iaa_sequence)
        dataset = dataset.map(
            lambda image, bboxes, image_height, image_width, labels: tuple([
                *tf.py_func(image_argument_partial, [image, bboxes], [image.dtype, bboxes.dtype]),
                image_height, image_width, labels])
        )


    preprocessing_partial_func = partial(preprocessing_training_func,
                                         min_size=min_size, max_size=max_size,
                                         preprocessing_type=preprocessing_type,
                                         caffe_pixel_means=caffe_pixel_means)

    dataset = dataset.batch(batch_size=batch_size).map(preprocessing_partial_func)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=prefecth_size)
    
    dataset = dataset.repeat(repeat_size)
    return dataset

# eval
def get_dataset_by_local_file(mode, root_path, image_format='bgr',
                              preprocessing_type='caffe', caffe_pixel_means=None,
                              min_edge=600, max_edge=1000):
    """
    根据 /path/to/VOC2007 or VOC2012/ImageSets/Main/{}.txt 读取图片列表，读取图片
    :param mode:
    :param root_path:
    :param image_format:
    :param caffe_pixel_means: 
    :param preprocessing_type:
    :param min_edge: 
    :param max_edge: 
    :return: 
    """
    if image_format not in ['rgb', 'bgr']:
        raise ValueError('unknown image format {}'.format(image_format))
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')

    def _map_from_cv2(example):
        example = example.decode()
        img_file_path = os.path.join(img_dir, example + '.jpg')
        img = cv2.imread(img_file_path).astype(np.float32)
        if preprocessing_type == 'caffe':
            img -= np.array([[caffe_pixel_means]])
        elif preprocessing_type == 'tf':
            img = img / 255.0 * 2.0 - 1.0
        else:
            raise ValueError('unknown preprocessing type {}'.format(preprocessing_type))
        h, w, _ = img.shape
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        new_h = int(scale * h)
        new_w = int(scale * w)

        img = cv2.resize(img, (new_w, new_h))
        if image_format == 'rgb':
            img = img[..., ::-1]
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(examples_list).map(
        lambda example: tf.py_func(_map_from_cv2,
                                   [example],
                                   [tf.float32, tf.float64, tf.int64, tf.int64]  # linux
                                   # [tf.float32, tf.float64, tf.int32, tf.int32]  # windows
                                   )
    ).batch(1)

    return dataset, examples_list


def _caffe_preprocessing(image, pixel_means):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 BGR 格式，并减去 imagenet 平均数
    :param image:
    :return:
    """
    image = tf.to_float(image)
    image = tf.reverse(image, axis=[-1])
    channels = tf.split(axis=-1, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] -= pixel_means[i]
    return tf.concat(axis=-1, values=channels)


def _tf_preprocessing(image):
    """
    输入 uint8 RGB 的图像，转换为 tf.float32 RGB 格式，取值范围[-1, 1]
    :param image:
    :return:
    """
    return tf.image.convert_image_dtype(image, dtype=tf.float32) * 2.0 - 1.0


def get_dataset_by_tf_records(mode, root_path,
                              preprocessing_type='caffe', caffe_pixel_means=None,
                              min_edge=600, max_edge=1000):
    with open(os.path.join(root_path, 'ImageSets', 'Main', '%s.txt' % mode), 'r') as f:
        lines = f.readlines()
    examples_list = [line.strip() for line in lines]
    img_dir = os.path.join(root_path, 'JPEGImages')
    example_path_list = [os.path.join(img_dir, example+'.jpg') for example in examples_list]

    def _map_from_tf_image(example_path):
        img = tf.image.decode_jpeg(tf.read_file(example_path), channels=3)
        if preprocessing_type == 'caffe':
            preprocessing_fn = partial(_caffe_preprocessing, pixel_means=caffe_pixel_means)
        elif preprocessing_type == 'tf':
            preprocessing_fn = _tf_preprocessing
        else:
            raise ValueError('unknown preprocessing type {}'.format(preprocessing_type))
        img = preprocessing_fn(img)

        # TODO: could not get image shape
        h, w, _ = img.get_shape().as_list()
        scale1 = min_edge / min(h, w)
        scale2 = max_edge / max(h, w)
        scale = min(scale1, scale2)
        img = tf.image.resize_bilinear(img, [tf.to_int32(scale*h), tf.to_int32(scale*w)])
        return img, float(scale), h, w

    dataset = tf.data.Dataset.from_tensor_slices(example_path_list).map(_map_from_tf_image).batch(1)

    return dataset, examples_list