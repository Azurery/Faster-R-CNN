import tensorflow as tf
from keras.layers import Concatenate, MaxPooling2D

class roi_pooling_creator(tf.keras.Model):
    
    def __init__(self, roi_pooling_size, roi_pooling_max_pooling=True):
        super().__init__()
        self._roi_pooling_size = roi_pooling_size
        self._roi_pooling_max_pooling = roi_pooling_max_pooling
        # self._concat = Concatenate(axis=0)
        self._max_pooling = MaxPooling2D(padding='same') 

    
    def call(self, inputs, training=None):
        '''
        输入：
            -经过 backbone 的 feature maps
            -经过 RPN 的 proposals
        输出：
            在 feature maps 上，对每一个 proposal 获取一个固定尺寸的 feature map
        
        RoI Pooling 是一种特殊的 Pooling 操作，给定一张图片的 feature map (`512 × H/16 × W/16`) ，和 `128` 个候选区域
        的坐标（`128 × 4`），RoI Pooling 将这些区域统一下采样到 （`512 × 7 × 7`），就得到了 `128 × 512 × 7 × 7` 的向量。
        可以看成是一个 `batch-size = 128`，通道数为 `512`，`7×7` 的 feature map。

        为什么要 pooling 成 `7×7` 的尺度？
            是为了能够共享权重。除了用到 VGG 前几层的卷积之外，最后的全连接层也可以继续利用。当所有的 RoIs 都被 pooling 成（ `512 × 7 × 7`）的
        feature map 后，将它 reshape 成一个一维的向量，就可以利用 VGG16 预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：
            -fc 21 用来分类，预测 RoIs 属于哪个类别（`20` 个类 + 背景）
            -fc 84 用来回归位置（`21` 个类，每个类都有 `4` 个位置参数）
        '''

        # 其中， feature_maps 的 shape：`[batch_size, height, width, channels]`
        #       proposals（即 RoIs）的 shape： `[num_proposals, 4]`
        feature_maps, proposals, spatial_scale = inputs
        proposals = proposals / tf.cast(spatial_scale, tf.float32)

        # `tf.shape(proposals)[0]` 表示 proposals 的索引
        bbox_indices = tf.zeros([tf.shape(proposals)[0]], dtype=tf.int32)

        h, w = feature_maps.get_shape().as_list()[1:3]
        proposal_channels = tf.split(proposals, 4, axis=1)

        # `boxes`：指需要划分的区域，输入格式为 `[[ymin, xmin, ymax, xmax]]` (要注意！这是一个二维列表)。
        # 且 `bboxes` 的坐标是相对于图像的宽高，即值介于 `(0, 1)`
        bboxes = tf.concat([proposal_channels[1] / tf.cast(h, dtype=tf.float32),
                            proposal_channels[0] / tf.cast(w, dtype=tf.float32),
                            proposal_channels[3] / tf.cast(h, dtype=tf.float32),
                            proposal_channels[2] / tf.cast(w, dtype=tf.float32)], axis=1)

        if self._roi_pooling_max_pooling:
            before_pooling_size = self._roi_pooling_size * 2

            # 【注】feature_maps 是需要参与反向传播的，但 bboxes 不需要参加
            # 现在如果我不加上 `tf.stop_gradient(bboxes)`

            # 投入的是 tensor，那么肯定不止一张图片啦，`box_ind` 这个参数就是为了索引用的。
            crops = tf.image.crop_and_resize(feature_maps, boxes=bboxes,
                                                            box_ind=bbox_indices,
                                                            crop_size=[before_pooling_size, before_pooling_size],
                                                            name='crops')
            return self._max_pooling(crops)
        else:
            crops = tf.image.crop_and_resize(feature_maps, boxes=bboxes,
                                                            box_ind=bbox_indices,
                                                            crop_size=[self._roi_pooling_size, self._roi_pooling_size],
                                                            name='crops')
            return crops

class roi_pooling_fpn_creator(tf.keras.Model):
    def __init__(self, roi_pooling_size):
        super().__init__()
        self._roi_pooling_size = roi_pooling_size
        self._max_pooling = MaxPooling2D(padding='same')

    def call(self, inputs, training=None):
        feature_maps, proposals, image_shape = inputs
        h, w = tf.cast(image_shape[0], dtype=tf.float32), tf.cast(image_shape[1], dtype=tf.float32)
        
        bbox_indices = tf.zeros([tf.shape(proposals)[0]], dtype=tf.int32)
        proposal_channels = tf.split(proposals, 4, axis=1)
        # 个人感觉有问题
        bboxes = tf.concat([proposal_channels[1] / tf.cast(h, dtype=tf.float32),
                            proposal_channels[0] / tf.cast(w, dtype=tf.float32),
                            proposal_channels[3] / tf.cast(h, dtype=tf.float32),
                            proposal_channels[2] / tf.cast(w, dtype=tf.float32)], axis=1)
        
        before_pooling_size = self._roi_pooling_size * 2
        crops = tf.image.crop_and_resize(feature_maps, boxes=bboxes, crop_size=[before_pooling_size, before_pooling_size],
                                        box_ind=bbox_indices, name='crops_resnet')
        return self._max_pooling(crops)
