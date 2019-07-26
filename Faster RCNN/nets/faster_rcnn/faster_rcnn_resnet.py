import tensorflow as tf
from keras.regularizers import l2
from nets.faster_rcnn.faster_rcnn_base import faster_rcnn_base

layers = tf.keras.layers

RESNET_WEIGHTS_PATH = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def finetune_resnet(resnet, resnet_mode='rpn_resnet', model_name='resnet', weight_decay=0.0001):
    input = layers.Input(shape=(None, None, 3))
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_padding')(input)
    
    x = layers.Conv2D(64, 7, use_bias=True, name='conv1_conv', trainable=True, padding='valid', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name='conv1_bn', trainable=False)(x, training=False)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_padding')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pooling')(x)

    x = resnet(x)
    model = tf.keras.Model(inputs=input, outputs=x, name=model_name)
    model.load_weights(RESNET_WEIGHTS_PATH, by_name=True)
    tf.logging.info('successfully load keras pre-trained weights for {} extractor.'.format(model_name))
    return model
    
def residual_block(input, num_filters, 
                filter_size=3, stride=1, 
                conv_shortcut=True, name=None,
                trainable=True, weight_decay=0.0001):
    if conv_shortcut is True:
        '''
        由于没有考虑resnet-34,所以直接是256
        '''
        shortcut = layers.Conv2D(4 * num_filters, 1, strides=stride, name=name + '_conv0', trainable=trainable, kernel_regularizer=l2(weight_decay))(input)
        shortcut = layers.BatchNormalization(axis=3, epsilon=1e-5, name=name + 'bn_0', trainable=False)(shortcut, training=False)
    else:
        shortcut = input
    
    x = layers.Conv2D(num_filters, 1, strides=stride, name=name + '_conv1', trainable=trainable, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(input)
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name=name + 'bn_1', trainable=False)(x, training=False)
    x = layers.Activation('relu', name=name + '_relu1')(x)

    x = layers.Conv2D(num_filters, filter_size, padding='same', name=name + '_conv2', trainable=trainable, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name=name + 'bn_2', trainable=False)(x, training=False)
    x = layers.Activation('relu', name=name + '_relu2')(x)

    x = layers.Conv2D(4 * num_filters, 1, name=name + '_conv3', trainable=trainable, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name=name + 'bn_3', trainable=False)(x, training=False)
    

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def bottleneck(input, num_filters, blocks, stride=2, name=None, trainable=True, weight_decay=0.0001):
    x = residual_block(input, num_filters, stride=stride, name=name + '_residual_block1', trainable=trainable, weight_decay=weight_decay)

    for i in range(2, blocks + 1):
        x = residual_block(x, num_filters, conv_shortcut=False, name=name + '_block' + str(i), trainable=trainable, weight_decay=weight_decay)
    
    return x 

# def feature_extractor_resnet_creator(depth, weight_decay=0.0001):
#     if depth == 50:
#         def resnet(input):
#             c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
#             c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
#             c4 = bottleneck(c3, 256, 6, name='conv4', weight_decay=weight_decay)
#             c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
#             return c2, c3, c4, c5
#     elif depth == 101:
#         def resnet(input):
#             c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
#             c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
#             c4 = bottleneck(c3, 256, 23, name='conv4', weight_decay=weight_decay)
#             c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
#             return c2, c3, c4, c5
#     elif depth == 152:
#         def resnet(input):
#             c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
#             c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
#             c4 = bottleneck(c3, 256, 36, name='conv4', weight_decay=weight_decay)
#             c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
#             return c2, c3, c4, c5
#     else:
#         raise ValueError('unknown depth {}.'.format(depth))
    
#     return finetune_resnet(resnet, 'rpn_resnet')

class roi_head_resnet_creator(tf.keras.Model):
    def __init__(self, num_classes, roi_feature_size=(7, 7, 256), 
                keep_rate=0.5, weight_decay=0.0001):
        super().__init__()
        self._num_classes = num_classes

        self._fc1 = layers.Dense(1024, name='fc1', activation='relu', kernel_initializer=tf.random_normal_initializer(0, 0.01),
                        kernel_regularizer=l2(weight_decay), input_shape=[roi_feature_size])
        self._dropout1 = layers.Dropout(rate=1 - keep_rate)
        self._fc2 = layers.Dense(1024, name='fc2', activation='relu', kernel_initializer=tf.random_normal_initializer(0, 0.01),
                        kernel_regularizer=l2(weight_decay))
        self._dropout2 = layers.Dropout(rate=1 - keep_rate)

        self._roi_score_layer = layers.Dense(num_classes, name='roi_head_resnet_score', activation=None,
                                    kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                    kernel_regularizer=l2(weight_decay))
        self._roi_coordinate_layer = layers.Dense(4 * num_classes, name='roi_head_resnet_coordinate',
                                        kernel_initializer=tf.random_normal_initializer(0, 0.01),
                                        kernel_regularizer=l2(weight_decay))
        self._flatten_layer = layers.Flatten()

    def call(self, inputs, training=None):
        x = self._flatten_layer(inputs)
        x = self._fc1(x)
        x = self._fc2(x)
        roi_scores = self._roi_score_layer(x)
        roi_coordinates = self._roi_coordinate_layer(x)
        return roi_scores, roi_coordinates


def feature_extractor_resnet_creator(depth, weight_decay=0.0001):
    # print(layers.Input(shape=(None, None, 3)))
    input = layers.Input(shape=(None, None, 3))
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_padding')(input)
    
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv', trainable=False, padding='valid')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name='conv1_bn', trainable=False)(x, training=False)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_padding')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pooling')(x)

    if depth == 50:
        def resnet(input):
            x = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=False, weight_decay=weight_decay)
            x = bottleneck(x, 128, 4, name='conv3', weight_decay=weight_decay)
            x = bottleneck(x, 256, 6, name='conv4', weight_decay=weight_decay)
            return x
    elif depth == 101:
        def resnet(input):
            x = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=False, weight_decay=weight_decay)
            x = bottleneck(x, 128, 4, name='conv3', weight_decay=weight_decay)
            x = bottleneck(x, 256, 23, name='conv4', weight_decay=weight_decay)
            return x
    elif depth == 152:
        def resnet(input):
            x = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=False, weight_decay=weight_decay)
            x = bottleneck(x, 128, 4, name='conv3', weight_decay=weight_decay)
            x = bottleneck(x, 256, 36, name='conv4', weight_decay=weight_decay)
            return x
    else:
        raise ValueError('unknown depth {}.'.format(depth))
    
    return finetune_resnet(resnet, 'rpn_resnet')
    

class faster_rcnn_resnet(faster_rcnn_base):
    def __init__(self,
                depth=50,
                roi_head_keep_dropout_rate=0.5,
                roi_feature_size=(7, 7, 1024),

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
        self._depth = depth
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
        return roi_head_resnet_creator(num_classes=self._num_classes, roi_feature_size=self._roi_feature_size)

    def feature_extractor_creator(self):
        return feature_extractor_resnet_creator(depth=self._depth)

