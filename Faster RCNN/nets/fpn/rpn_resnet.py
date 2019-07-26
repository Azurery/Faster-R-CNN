import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D,\
                        Dense, Dropout, Flatten, UpSampling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from nets.fpn_base import fpn_base

layers = tf.keras.layers

# WEIGHTS_PATH = ('https://github.com/keras-team/keras-applications/releases/download/resnet/')
# WEIGHTS_HASHES = {
#     'resnet50': ('2cb95161c43110f7111970584f804107',
#                 '4d473c1dd8becc155b73f8504c6f6626'),
#     'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
#                 '88cf7a10940856eca736dc7b7e228a21'),
#     'resnet152': ('100835be76be38e30d865e96f2aaae62',
#                 'ee4c566cf9a93f14d82f913c2dc6dd0c'),
#     'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
#                 'fac2f116257151a9d068a22e544a4917'),
#     'resnet101v2': ('6343647c601c52e1368623803854d971',
#                     'c0ed64b8031c3730f411d2eb4eea35b5'),
#     'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
#                     'ed17cf2e0169df9d443503ef94b23b33'),
#     'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
#                 '62527c363bdd9ec598bed41947b379fc'),
#     'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
#                 '0f678c91647380debd923963594981b3')
# }

RESNET_WEIGHTS_PATH = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def finetune_resnet(resnet, resnet_mode='rpn_resnet', model_name='resnet', weight_decay=0.0001):
    input = Input(shape=(None, None, 3))
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_padding')(input)
    
    x = Conv2D(64, 7, use_bias=True, name='conv1_conv', trainble=True, padding='valid', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3, epsilon=1e-5, name='conv1_bn', trainable=False)(x, training=False)
    x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_padding')(x)
    x = MaxPooling2D(3, stride=2, name='pool1_pooling')(x)

    if resnet_mode == 'rpn_resnet':
        c2, c3, c4, c5 = resnet(x)
        model = tf.keras.Model(inputs=input, outputs=[c2, c3, c4, c5], name=model_name)
    else:
        x = resnet(x)
        model = tf.keras.Model(inputs=input, outputs=x, name=model_name)
    # load weights
    # if model_name in WEIGHTS_HASHES:
    #     file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #     file_hash = WEIGHTS_HASHES[model_name][1]
    #     weights_path = tf.keras.utils.get_file(file_name, origin=WEIGHTS_PATH + file_name, cache_dir='../weights/', file_hash=file_hash)
    #     model.load_weights(weights_path, by_name=True)
    #     tf.logging.info('successfully load keras pre-trained weights for {} extractor.'.format(model_name))
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

def feature_extractor_resnet_creator(depth, weight_decay=0.0001):
    if depth == 50:
        def resnet(input):
            c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
            c4 = bottleneck(c3, 256, 6, name='conv4', weight_decay=weight_decay)
            c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    elif depth == 101:
        def resnet(input):
            c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
            c4 = bottleneck(c3, 256, 23, name='conv4', weight_decay=weight_decay)
            c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    elif depth == 152:
        def resnet(input):
            c2 = bottleneck(input, 64, 3, stride=1, name='conv2', trainable=True, weight_decay=weight_decay)
            c3 = bottleneck(c2, 128, 4, name='conv3', weight_decay=weight_decay)
            c4 = bottleneck(c3, 256, 36, name='conv4', weight_decay=weight_decay)
            c5 = bottleneck(c4, 512, 3, name='conv5', weight_decay=weight_decay)
            return c2, c3, c4, c5
    else:
        raise ValueError('unknown depth {}.'.format(depth))
    
    return finetune_resnet(resnet, 'rpn_resnet')
    
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

class plist_creator(tf.keras.Model):
    def __init__(self, top_down_channels=256, weight_decay=0.0001, use_bias=True):
        super().__init__()

        self._create_p5_conv = layers.Conv2D(top_down_channels, 1, strides=1, use_bias=use_bias, name='create_p5_conv', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
        self._create_p6_upsampling = layers.MaxPooling2D(strides=2, pool_size=(1, 1), name='create_p6')

        self._create_m4_conv = layers.Conv2D(top_down_channels, 1, strides=(1, 1), use_bias=use_bias, name='create_m4_conv', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
        self._create_m4_upsampling = layers.UpSampling2D(size=(2, 2), name='create_m4_upsampling')
        self._create_m4_fusion = layers.Add(name='create_m4_fusion')
        self._create_p4 = layers.Conv2D(top_down_channels, 3, 1, use_bias=use_bias, padding='same', name='create_p4', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
        
        self._create_m3_conv = layers.Conv2D(top_down_channels, 1, strides=(1, 1), use_bias=use_bias, name='create_m3_conv', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
        self._create_m3_upsampling = layers.UpSampling2D(size=(2, 2), name='create_m3_upsampling')
        self._create_m3_fusion = layers.Add(name='create_m3_fusion')
        self._create_p3 = layers.Conv2D(top_down_channels, 3, strides=(1, 1), use_bias=use_bias, name='create_p3', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))

        self._create_m2_conv = layers.Conv2D(top_down_channels, 1, strides=(1, 1), use_bias=use_bias, name='create_m2_conv', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))
        self._create_m2_upsampling = layers.UpSampling2D(size=(2, 2), name='create_m2_upsampling')        
        self._create_m2_fusion = layers.Add(name='create_m2_fusion')
        self._create_p2 = layers.Conv2D(top_down_channels, 3, strides=(1, 1), use_bias=use_bias, name='create_p2', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))

    def call(self, inputs, training=None):
        c2, c3, c4, c5 = inputs

        # create p5, p6
        p5 = self._create_p5_conv(c5)
        p6 = self._create_p6_upsampling(p5)

        # FIXME: m4_upsampling 存在问题，是否应该用 tf.image.resize_bilinear()
        # 在进行融合时，为什么要 *0.5？
        # create p4
        m4_conv = self._create_m4_conv(c4)
        m4_upsampling = self._create_m4_upsampling(p5)
        m4 = self._create_m4_fusion([m4_conv * 0.5, m4_upsampling * 0.5])

        # create p3
        m3_conv = self._create_m3_conv(c3)
        m3_upsampling = self._create_m3_upsampling(m4)
        m3 = self._create_m3_fusion([m3_conv * 0.5, m3_upsampling * 0.5])

        # create p2
        m2_conv = self._create_m2_conv(c2)
        m2_upsampling = self._create_m2_upsampling(m3)
        m2 = self._create_m2_fusion([m2_conv * 0.5, m2_upsampling * 0.5])

        p4 = self._create_p4(m4)
        p3 = self._create_p3(m3)
        p2 = self._create_p2(m2)
        return p2, p3, p4, p5, p6


class fpn_resnet(fpn_base):
    def __init__(self, 
                # fpn 参数
                depth=50,
                level_list = ('p2', 'p3', 'p4', 'p5', 'p6'),
                min_level = 2, 
                max_level = 5,
                roi_feature_size=(7, 7, 256),
                top_down_channels=256,

                num_classes=21,
                weight_decay=0.0001,

                # rpn anchors特有参数
                aspect_ratios=(0.5, 1.0, 2.0),
                base_anchor_sizes=(32, 64, 128, 256, 512),
                extractor_strides=(4, 8, 16, 32, 64),
                anchor_steps=(1.), 

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

                roi_head_keep_dropout_rate=0.5,
                roi_training_positive_iou_threshold=.5,
                roi_training_negative_iou_threshold=.1,
                roi_training_total_num_samples=128,
                roi_training_max_positive_samples=32):
        self._depth = depth
        self._roi_head_keep_dropout_rate = roi_head_keep_dropout_rate
        self._top_down_channels = top_down_channels
        super().__init__(level_list = ('p2', 'p3', 'p4', 'p5', 'p6'),
                min_level = min_level, 
                max_level = max_level,
                roi_feature_size=roi_feature_size,

                num_classes=num_classes,
                weight_decay=weight_decay,

                # rpn anchors特有参数
                aspect_ratios=aspect_ratios,
                base_anchor_sizes=base_anchor_sizes,
                extractor_strides=extractor_strides,
                anchor_steps=anchor_steps, 

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

                roi_head_keep_dropout_rate=roi_head_keep_dropout_rate,
                roi_training_positive_iou_threshold=roi_training_positive_iou_threshold,
                roi_training_negative_iou_threshold=roi_training_negative_iou_threshold,
                roi_training_total_num_samples=roi_training_total_num_samples,
                roi_training_max_positive_samples=roi_training_max_positive_samples)

    def roi_head_creator(self):
        return roi_head_resnet_creator(num_classes=self._num_classes, roi_feature_size=self._roi_feature_size,
                            keep_rate=self._roi_head_keep_dropout_rate, weight_decay=self._weight_decay)

    def feature_extractor_creator(self):
        return feature_extractor_resnet_creator(depth=self._depth, weight_decay=self._weight_decay)

    def plist_creator(self):
        return plist_creator(top_down_channels=self._top_down_channels, weight_decay=self._weight_decay)
    
    