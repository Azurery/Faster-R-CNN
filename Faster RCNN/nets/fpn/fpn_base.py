import tensorflow as tf

from utils.proposals_utils import rpn_head_creator, rpn_proposals_creator, rpn_target_anchors_createor, roi_target_proposals_createor
from utils.roi_pooling_utils import roi_pooling_creator, roi_pooling_fpn_creator
from utils.bbox_utils import generate_anchors
from utils.loss_utils import rpn_losses_creator, roi_losses_creator

class fpn_base(tf.keras.Model):
    def __init__(self,
                # fpn 参数
                level_list = ('p2', 'p3', 'p4', 'p5', 'p6'),
                min_level = 2, 
                max_level = 5,
                roi_feature_size=(7, 7, 256),

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

                roi_training_positive_iou_threshold=.5,
                roi_training_negative_iou_threshold=.1,
                roi_training_total_num_samples=128,
                roi_training_max_positive_samples=32):
        super().__init__()
        self._level_list = level_list
        self._min_level = min_level
        self._max_level = max_level
        self._roi_feature_size = roi_feature_size

        self._num_classes = num_classes
        self._weight_decay = weight_decay

        self._extractor_strides = extractor_strides        
        self._aspect_ratios = aspect_ratios
        self._base_anchor_sizes = base_anchor_sizes
        self._anchor_steps = anchor_steps
        self._num_anchors = len(self._aspect_ratios)    # 每个level只有３种anchors

        self._rpn_proposals_nms_threshold = rpn_proposals_nms_threshold
        self._rpn_proposals_num_pre_nms_train = rpn_proposals_num_pre_nms_train
        self._rpn_proposals_num_post_nms_train = rpn_proposals_num_post_nms_train
        self._rpn_proposals_num_pre_nms_test = rpn_proposals_num_pre_nms_test
        self._rpn_proposals_num_post_nms_test = rpn_proposals_num_post_nms_test


		# rpn target anchors 参数
        self._rpn_target_anchors_positive_iou_threshold = rpn_target_anchors_positive_iou_threshold
        self._rpn_target_anchors_negative_iou_threshold = rpn_target_anchors_negative_iou_threshold
        self._rpn_target_anchors_total_samples = rpn_target_anchors_total_samples
        self._rpn_target_anchors_max_positive_samples = rpn_target_anchors_max_positive_samples
									
		# roi pooling 参数
        self._roi_pooling_size = roi_pooling_size
        self._roi_pooling_max_pooling = roi_pooling_max_pooling

        self._roi_training_positive_iou_threshold = roi_training_positive_iou_threshold
        self._roi_training_negative_iou_threshold = roi_training_negative_iou_threshold
        self._roi_training_total_num_samples = roi_training_total_num_samples
        self._roi_training_max_positive_samples = roi_training_max_positive_samples


        # 创建 Faster R-CNN 的基础组件

		# 创建 Feature Extractor
        self._feature_extractor_creator = self.feature_extractor_creator()

        self._plist_creator = self.plist_creator()

        self._rpn_head_creator = rpn_head_creator(self._num_anchors, weight_decay=weight_decay)
    
        self._rpn_proposals_creator = rpn_proposals_creator(num_anchors=self._num_anchors,
                                                            nms_threshold=self._rpn_proposals_nms_threshold,
                                                            num_pre_nms_train=self._rpn_proposals_num_pre_nms_train,
                                                            num_post_nms_train=self._rpn_proposals_num_post_nms_train,
                                                            num_pre_nms_test=self._rpn_proposals_num_pre_nms_test,
                                                            num_post_nms_test=self._rpn_proposals_num_post_nms_test)

        self._rpn_target_anchors_creator = rpn_target_anchors_createor(
											positive_iou_threshold=self._rpn_target_anchors_positive_iou_threshold,
											negative_iou_threshold=self._rpn_target_anchors_negative_iou_threshold,
											total_samples=self._rpn_target_anchors_total_samples,
											max_positive_samples=self._rpn_target_anchors_max_positive_samples)

        # self._roi_pooling_creator = roi_pooling_creator(roi_pooling_size=roi_pooling_size)
        self._roi_head_creator = self.roi_head_creator()
        self._roi_target_proposals_createor = roi_target_proposals_createor(num_classes=self._num_classes, 
                                                                            positive_iou_threshold=self._roi_training_positive_iou_threshold,
                                                                            negative_iou_threshold=self._roi_training_negative_iou_threshold,
                                                                            num_samples=self._roi_training_total_num_samples,
                                                                            max_positive_samples=self._roi_training_max_positive_samples)

        self._roi_pooling = roi_pooling_fpn_creator(roi_pooling_size=roi_pooling_size)


    def feature_extractor_creator(self):
        raise NotImplementedError

    def roi_head_creator(self):
        raise NotImplementedError
    
    def plist_creator(self):
        raise NotImplementedError

    def call(self, inputs, training=None):
        if training:
            image, gt_bboxes, gt_labels = inputs
        else:
            image = inputs
        
        image_shape = image.get_shape().as_list()[1:3]

        clist = self._feature_extractor_creator(image, training=training)
        plist = self._plist_creator(clist, trianing=training)
        tf.logging.info('feature maps lengtj is {}.'.format(len(plist)))
        
        for index, p in enumerate(plist):
            tf.logging.info('p{} shape is {}.'.format(index + 1, p.get_shape().as_list()))

        rpn_scores, rpn_coordinates = self.total_rpn_head_creator(plist)
        anchors = self.generate_total_anchors(image_shape)
        rpn_scores = tf.nn.softmax(rpn_scores)[:, 1]
        
        rpn_proposals = self._rpn_proposals_creator((rpn_scores, rpn_coordinates, anchors, image_shape), 
                                                training=training)

        if training:
            rpn_labels, rpn_target_anchors, rpn_w_in, rpn_w_out = \
                    self._rpn_target_anchors_creator((gt_bboxes, image_shape, anchors), training=training)
            
            rpn_cls_loss, rpn_reg_loss = rpn_losses_creator(self._num_anchors, rpn_scores, rpn_target_anchors, 
                                                            rpn_labels, rpn_target_anchors, rpn_w_in, rpn_w_out, tag='faster_rcnn_fpn')

            roi_proposals, roi_labels, roi_target_bboxes, roi_w_in, roi_w_out = \
                    self._roi_target_proposals_createor((rpn_proposals, gt_bboxes, gt_labels), training=training)

            roi_features, selected_roi_index = self.select_roi_features(roi_proposals, plist, image_shape)

            roi_scores, roi_coordinates = self._roi_head_creator()

            roi_labels = tf.gather(roi_labels, selected_roi_index)
            roi_coordinates = tf.gather(roi_coordinates, selected_roi_index)
            roi_w_in = tf.gather(roi_w_in, selected_roi_index)
            roi_w_out = tf.gather(roi_w_out, selected_roi_index)

            roi_cls_loss, roi_reg_loss = roi_losses_creator(roi_scores, roi_coordinates, 
                                                            roi_labels, roi_target_bboxes,
                                                            roi_w_in, roi_w_out)

            return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss
        # else:

    def total_rpn_head_creator(self, plist):
        total_rpn_scores = []
        total_rpn_coordinates = []
        for level, p in zip(self._level_list, plist):
            rpn_scores, rpn_coordinates = self._rpn_head_creator(p)
            total_rpn_scores.append(rpn_scores)
            total_rpn_coordinates.append(rpn_coordinates)
            tf.logging.info('fpn {} rpn score shape {}.'.format(level, rpn_scores.get_shape().as_list()))
        
        total_rpn_scores = tf.concat(total_rpn_scores, axis=0, name='fpn_total_rpn_scores')
        total_rpn_coordinates = tf.concat(total_rpn_coordinates, axis=0, name='fpn_total_rpn_coordinates')
        tf.logging.info('fpn_total_scores shape is {}.'.format(total_rpn_scores.get_shape().as_list()))
        tf.logging.info('fpn_total_coordinates shape is {}.'.format(total_rpn_coordinates.get_shape().as_list()))

        return total_rpn_scores, total_rpn_coordinates

    def generate_total_anchors(self, image_shape):
        total_anchors = []
        for index in range(len(self._level_list)):
            level_name = self._level_list[index]
            extracor_stride = self._extractor_strides[index]

            anchors = generate_anchors(image_shape[0], image_shape[1], base_anchor_size=self._base_anchor_sizes[index],
                                    feature_stride=extracor_stride, anchor_ratios=self._aspect_ratios, anchor_steps=self._anchor_steps)
            total_anchors.append(anchors)
            tf.logging.info('{} generates {} anchors.'.format(level_name, anchors.shape[0]))

        total_anchors = tf.concat(total_anchors, axis=0, name='total_anchors')
        tf.logging.info('total_anchors shape is {}.'.format(total_anchors.get_shape().as_list()))
        return total_anchors

    def select_roi_features(self, final_roi_proposals, plist, image_shape):
        xmin, ymin, xmax, ymax = tf.unstack(final_roi_proposals, axis=1)
        h = tf.maximum(ymax - ymin, 0)
        w = tf.maximum(xmax - xmin, 0)
        roi_levels = tf.floor(4. + tf.log(tf.sqrt(h * w + 1e-8) / 224) / tf.log(2.))

        # 设置level的上下限
        roi_levels = tf.maximum(roi_levels, tf.ones_like(roi_levels) * self._min_level)
        roi_levels = tf.minimum(roi_levels, tf.ones_like(roi_levels) * self._max_level)
        roi_levels = tf.stop_gradient(tf.reshape(roi_levels, [-1]))

        rois2level = []
        total_rois = []
        for i in range(self._min_level, self._max_level + 1):
            # 先找出需要在第level层计算roi
            level_index = tf.where(tf.equal(roi_levels, i))
            level_rois = tf.stop_gradient(tf.gather(final_roi_proposals, level_index))

            total_rois.append(level_rois)
            rois2level.append(level_index)
        
        selected_roi_index = tf.concat(rois2level, axis=0, name='selected_roi_index')

        total_roi_features = []
        for level_name, rois, p, stride in zip(self._level_list[:1], total_rois, plist, self._extractor_strides):
            if rois.shape[0] == 0:
                continue
            roi_features = self._roi_pooling()
            total_roi_features.append(roi_features)
            tf.logging.info('{} generates {} roi features.'.format(level_name, roi_features.shape[0]))
        return tf.concat(total_roi_features, axis=0, name='total_roi_features'), selected_roi_index
            



