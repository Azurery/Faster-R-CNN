import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
from nets.faster_rcnn.faster_rcnn_base import faster_rcnn_base

layers = tf.keras.layers

VGG16_WEIGHTS_PATH = 'weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

class faster_rcnn_vgg16(faster_rcnn_base):
    def __init__(self,
                # faster_rcnn_vgg16特有的参数
                slim_ckpt_file_path=None,
                roi_head_keep_dropout_rate=0.5,
                roi_feature_size=(7, 7, 512),

                num_classes=21,
                weight_decay=0.0001,
                aspect_ratios=(0.5, 1, 2),
                anchor_steps=(8, 16, 32),
                feature_stride=16,

                # rpn proposals 参数
                rpn_proposals_nms_threshold=0.7,
                rpn_proposals_num_pre_nms_train=12000,
                rpn_proposals_num_post_nms_train=2000,
                rpn_proposals_num_pre_nms_test=6000,
                rpn_proposals_num_post_nms_test=300,

                # rpn target anchors 参数
                rpn_target_anchors_positive_iou_threshold=0.7,
                rpn_target_anchors_negative_iou_threshold=0.3,
                rpn_target_anchors_total_samples=256,
                rpn_target_anchors_max_positive_samples=128,

                # roi pooling 参数
                roi_pooling_size=7,
                roi_pooling_max_pooling=True,

                roi_training_positive_iou_threshold=.5,
                roi_training_negative_iou_threshold=.1,
                roi_training_total_num_samples=128,
                roi_training_max_positive_samples=32):
        self._slim_ckpt_file_path = slim_ckpt_file_path
        self._roi_head_keep_dropout_rate = roi_head_keep_dropout_rate
        self._roi_feature_size = roi_feature_size
        super().__init__(num_classes=num_classes,
                        weight_decay=weight_decay,
                        aspect_ratios=aspect_ratios,
                        anchor_steps=anchor_steps,
                        feature_stride=feature_stride,

                        # rpn proposals 参数
                        rpn_proposals_nms_threshold=rpn_proposals_nms_threshold,
                        rpn_proposals_num_pre_nms_train=rpn_proposals_num_pre_nms_train,
                        rpn_proposals_num_post_nms_train=rpn_proposals_num_post_nms_train,
                        rpn_proposals_num_pre_nms_test=rpn_proposals_num_pre_nms_test,
                        rpn_proposals_num_post_nms_test=rpn_proposals_num_post_nms_test,

                        # rpn target anchors 参数
                        rpn_target_anchors_positive_iou_threshold=rpn_target_anchors_positive_iou_threshold,
                        rpn_target_anchors_negative_iou_threshold=rpn_target_anchors_negative_iou_threshold,
                        rpn_target_anchors_total_samples=rpn_target_anchors_total_samples,
                        rpn_target_anchors_max_positive_samples=rpn_target_anchors_max_positive_samples,

                        # roi pooling 参数
                        roi_pooling_size=roi_pooling_size,
                        roi_pooling_max_pooling=roi_pooling_max_pooling,

                        roi_training_positive_iou_threshold=roi_training_positive_iou_threshold,
                        roi_training_negative_iou_threshold=roi_training_negative_iou_threshold,
                        roi_training_total_num_samples=roi_training_total_num_samples,
                        roi_training_max_positive_samples=roi_training_max_positive_samples, )

    def roi_head_creator(self):
        return roi_head_vgg16(self._num_classes,
                            roi_feature_size=self._roi_feature_size,
                            keep_rate=self._roi_head_keep_dropout_rate,
                            weight_decay=self._weight_decay,
                            slim_ckpt_file_path=self._slim_ckpt_file_path)

    def feature_extractor_creator(self):
        return feature_extractor_creator_vgg16(weight_decay=self._weight_decay,
                                            slim_ckpt_file_path=self._slim_ckpt_file_path)


class feature_extractor_creator_vgg16(tf.keras.Sequential):
    def __init__(self, weight_decay=0.0001, slim_ckpt_file_path=None):
        super().__init__(name='feature_extractor_creator_vgg16')
        # block1
        self.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False, kernel_regularizer=l2(weight_decay), input_shape=(None, None, 3)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.MaxPooling2D((2, 2), name='block1_pool', padding='same'))

        # block2
        self.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.MaxPooling2D((2, 2), name='block2_pool', padding='same'))

        # block3
        self.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.MaxPooling2D((2, 2), name='block3_pool', padding='same'))

        # block4
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.MaxPooling2D((2, 2), name='block4_pool', padding='same'))

        # block5
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False, kernel_regularizer=l2(weight_decay), input_shape=(None, None, 3)))
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False, kernel_regularizer=l2(weight_decay)))
        self.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False, kernel_regularizer=l2(weight_decay)))
        # print(self)
        if slim_ckpt_file_path:
            self._load_slim_weights(slim_ckpt_file_path)
        else:
            self._load_keras_weights(VGG16_WEIGHTS_PATH)

    def _load_keras_weights(self, weights_path):
        self.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully loaded keras vgg weights for vgg16 extractor.')

    def _load_slim_weights(self, slim_ckpt_file_path):
        reader = tf.train.NewCheckpointReader(slim_ckpt_file_path)
        slim_to_keras = {
            "vgg_16/conv1/conv1_1/": "block1_conv1",
            "vgg_16/conv1/conv1_2/": "block1_conv2",

            "vgg_16/conv2/conv2_1/": "block2_conv1",
            "vgg_16/conv2/conv2_2/": "block2_conv2",

            "vgg_16/conv3/conv3_1/": "block3_conv1",
            "vgg_16/conv3/conv3_2/": "block3_conv2",
            "vgg_16/conv3/conv3_3/": "block3_conv3",

            "vgg_16/conv4/conv4_1/": "block4_conv1",
            "vgg_16/conv4/conv4_2/": "block4_conv2",
            "vgg_16/conv4/conv4_3/": "block4_conv3",

            "vgg_16/conv5/conv5_1/": "block5_conv1",
            "vgg_16/conv5/conv5_2/": "block5_conv2",
            "vgg_16/conv5/conv5_3/": "block5_conv3",
        }
        for slim_tensor_name_pre in slim_to_keras.keys():
            if slim_tensor_name_pre == 'vgg_16/conv1/conv1_1/':
                weights = reader.get_tensor(slim_tensor_name_pre + 'weights')[:, :, ::-1, :]
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    weights,
                    reader.get_tensor(slim_tensor_name_pre + 'biases'),
                ])
            else:
                self.get_layer(name=slim_to_keras[slim_tensor_name_pre]).set_weights([
                    reader.get_tensor(slim_tensor_name_pre + 'weights'),
                    reader.get_tensor(slim_tensor_name_pre + 'biases'),
                ])
        tf.logging.info('successfully loaded slim vgg weights for vgg16 extractor.')


class roi_head_vgg16(tf.keras.Model):
    def __init__(self, num_classes, roi_feature_size=(7, 7, 512),
                keep_rate=0.5,
                weight_decay=0.00005,
                slim_ckpt_file_path=None, ):
        super().__init__()
        self._num_classes = num_classes

        self._fc1 = layers.Dense(4096, name='fc1', activation='relu',
                                kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                kernel_regularizer=l2(weight_decay),
                                input_shape=[roi_feature_size])

        self._dropout1 = layers.Dropout(rate=1 - keep_rate)
        self._fc2 = layers.Dense(4096, name='fc2', activation='relu',
                                kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                kernel_regularizer=l2(weight_decay))
        self._dropout2 = layers.Dropout(rate=1 - keep_rate)

        # 1*1 conv可以用fc layer 来代替，作用是一样的
        # 只是这里感觉有一个BUG,_roi_score_layer 应该使用 2*num_classes
        self._roi_score_layer = layers.Dense(num_classes, name='roi_head_vgg16_score',
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                            kernel_regularizer=l2(weight_decay))
        self._roi_coordinate_layer = layers.Dense(4 * num_classes, name='roi_head_vgg16_coordinate',
                                                kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                                kernel_regularizer=l2(weight_decay))
        self._flatten_layer = layers.Flatten()
        self.build((None, *roi_feature_size))

        if slim_ckpt_file_path is None:
            self._load_keras_weights(VGG16_WEIGHTS_PATH)
        else:
            self._load_slim_weights(slim_ckpt_file_path)

    def _load_slim_weights(self, ckpt_file_path):
        reader = tf.train.NewCheckpointReader(ckpt_file_path)
        slim_to_keras = {
            "vgg_16/fc6/": "fc1",
            "vgg_16/fc7/": "fc2",
        }

        for slim_tensor_name_pre in slim_to_keras.keys():
            cur_layer = self.get_layer(name=slim_to_keras[slim_tensor_name_pre])
            cur_layer.set_weights([
                reader.get_tensor(slim_tensor_name_pre + 'weights').reshape(
                    cur_layer.variables[0].get_shape().as_list()),
                reader.get_tensor(slim_tensor_name_pre + 'biases').reshape(
                    cur_layer.variables[1].get_shape().as_list()),
            ])
        tf.logging.info('successfully loaded slim vgg weights for roi head.')

    def _load_keras_weights(self, weights_path):
        self.load_weights(weights_path, by_name=True)
        tf.logging.info('successfully load pretrained weights for roi head.')

    def call(self, inputs, training=None):
        x = self._flatten_layer(inputs)
        x = self._fc1(x)
        x = self._dropout1(x, training)
        x = self._fc2(x)
        x = self._dropout2(x, training)
        roi_scores = self._roi_score_layer(x)
        roi_coordinates = self._roi_coordinate_layer(x)
        return roi_scores, roi_coordinates
