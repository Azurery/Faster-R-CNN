import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_bboxes_with_labels2image(image, bboxes, labels):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(bboxes, tf.Tensor):
        bboxes = bboxes.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    index = 0
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if labels is not None:
            cv2.putText(image, text=str(labels[index]), org=(xmin, ymin + 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1e-3 * image.shape[0], color=(0, 0, 255), thickness=2,)
        index += 1
    return image

def show_image(preprocessed_image, bboxes, labels, preprocessing_type='caffe', 
            caffe_pixel_means=None, figsize=(15, 10), enable_matplotlib=True):
    if isinstance(preprocessed_image, tf.Tensor):
        preprocessed_image = tf.squeeze(preprocessed_image, axis=0).numpy()
    if isinstance(bboxes, tf.Tensor):
        bboxes = bboxes.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()
    if preprocessing_type == 'caffe':
        cur_means = caffe_pixel_means
        preprocessed_image[..., 0] += cur_means[0]
        preprocessed_image[..., 1] += cur_means[1]
        preprocessed_image[..., 2] += cur_means[2]
        preprocessed_image = preprocessed_image[..., ::-1]
        preprocessed_image = preprocessed_image.astype(np.uint8)
    elif preprocessing_type == 'tensorflow':
        preprocessed_image = ((preprocessed_image + 1.0) / 2.0) * 255.0
        preprocessed_image = preprocessed_image.astype(np.uint8)
    elif preprocessing_type is None:
        pass
    else:
        raise ValueError('unknown preprocess_type {}.'.format(preprocessing_type))
    image_with_bboxes_labels = draw_bboxes_with_labels2image(preprocessed_image, bboxes, labels)
    if enable_matplotlib:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_bboxes_labels)
        plt.show()
    
    return image_with_bboxes_labels

    