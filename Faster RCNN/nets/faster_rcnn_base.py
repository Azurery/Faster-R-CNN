import numpy as np
import cv2
from collections import namedtuple
import tensorflow as tf
from keras.layers import Conv2D
from keras.regularizers import l2
from utils.proposals_utils import rpn_head_creator, rpn_proposals_creator, rpn_target_anchors_createor, roi_target_proposals_createor
from utils.roi_pooling_utils import roi_pooling_creator
from utils.bbox_utils import generate_anchors
from utils.loss_utils import rpn_losses_creator, roi_losses_creator

# learning rate flags
tf.app.flags.DEFINE_float(
	'weight_decay', 0.0001, 'The weight decay on the model weights.')


FLAGS = tf.app.flags.FLAGS

faster_rcnn_parameters = namedtuple('faster_rcnn_parameters',
									['num_classes',
									'weight_decay',
									'aspect_ratios',
									'anchor_sizes',
									'feature_stride',

									# rpn proposals 参数
									'rpn_proposals_nms_threshold',
									'rpn_proposals_num_pre_nms_train',
									'rpn_proposals_num_post_nms_train',
									'rpn_proposals_num_pre_nms_test',
									'rpn_proposals_num_post_nms_test',

									# rpn target anchors 参数
									'rpn_target_anchors_positive_iou_threshold',
									'rpn_target_anchors_negative_iou_threshold',
									'rpn_target_anchors_total_samples',
									'rpn_target_anchors_max_positive_samples',
									
									# roi pooling 参数
									'roi_pooling_size',
									'roi_pooling_max_pooling',

									'roi_training_positive_iou_threshold',
									'roi_training_negative_iou_threshold',
									'roi_training_total_num_samples',
									'roi_training_max_positive_samples'

									# prediction 参数
									])


class faster_rcnn_base(tf.keras.Model):
	def __init__(self, num_classes, weight_decay, aspect_ratios, anchor_sizes,feature_stride, 
				rpn_proposals_nms_threshold, 
				rpn_proposals_num_pre_nms_train,
				rpn_proposals_num_post_nms_train, 
				rpn_proposals_num_pre_nms_test, 
				rpn_proposals_num_post_nms_test,
				rpn_target_anchors_positive_iou_threshold, 
				rpn_target_anchors_negative_iou_threshold, 
				rpn_target_anchors_total_samples, 
				rpn_target_anchors_max_positive_samples,
				roi_pooling_size, 
				roi_pooling_max_pooling, 
				roi_training_positive_iou_threshold,
                roi_training_negative_iou_threshold,
                roi_training_total_num_samples,
                roi_training_max_positive_samples):
		super().__init__()
		self._num_classes = num_classes
		self._weight_decay = weight_decay
		self._aspect_ratios = aspect_ratios
		self._anchor_sizes = anchor_sizes
		self._feature_stride = feature_stride
		self._num_anchors = len(self._aspect_ratios) * len(self._anchor_sizes)

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

		# 1. 创建 RPN Head
		self._rpn_head_creator = rpn_head_creator(num_anchors=self._num_anchors,
												weight_decay=self._weight_decay)

		# 2. 创建 RPN Proposals
		# 在 RPN 中，从上万个 anchors 中，选择一定数目（`2000` 或者 `300`），调整大小和位置，生成 RoIs，
		# 用以 Faster R-CNN 训练或者测试。
		self._rpn_proposals_creator = rpn_proposals_creator(num_anchors=self._num_anchors,
													nms_threshold=self._rpn_proposals_nms_threshold,
													num_pre_nms_train=self._rpn_proposals_num_pre_nms_train,
													num_post_nms_train=self._rpn_proposals_num_post_nms_train,
													num_pre_nms_test=self._rpn_proposals_num_pre_nms_test,
													num_post_nms_test=self._rpn_proposals_num_post_nms_test)
		
		# 3. 创建 RPN Target Anchors
		# 从 `20000` 多个候选的 anchors 选出 `256` 个 anchors 进行分类和回归位置。
		self._rpn_target_anchors_creator = rpn_target_anchors_createor(
											positive_iou_threshold=self._rpn_target_anchors_positive_iou_threshold,
											negative_iou_threshold=self._rpn_target_anchors_negative_iou_threshold,
											total_samples=self._rpn_target_anchors_total_samples,
											max_positive_samples=self._rpn_target_anchors_max_positive_samples)

		# 创建 RoI Head
		self._roi_head_creator = self.roi_head_creator()

		# 4. 进行 RoI Pooling
		self._roi_pooling_creator = roi_pooling_creator(
								roi_pooling_size=self._roi_pooling_size,
								roi_pooling_max_pooling=self._roi_pooling_max_pooling)				

		# 5. 挑选出一些rois,进行roi训练
		self._roi_target_proposals_creator = roi_target_proposals_createor(num_classes=self._num_classes, 
																		positive_iou_threshold=self._roi_training_positive_iou_threshold,
																		negative_iou_threshold=self._roi_training_negative_iou_threshold,
																		num_samples=self._roi_training_total_num_samples,
																		max_positive_samples=self._roi_training_max_positive_samples)
	
	def call(self, inputs, training=None):
		if training:
			image, gt_bboxes, gt_labels = inputs
		else:
			image = inputs

		image_shape = image.get_shape().as_list()[1:3]
		# print('image_shape', image_shape)
		
		# tf.logging.debug('Image shape is {}'.format(image_shape))

		feature_maps = self._feature_extractor_creator(image, training=training)
		self._feature_maps_shape = feature_maps.get_shape().as_list()[1:3]
		# tf.logging.debug('Feature maps shape is {}'.format(feature_maps_shape))

		# FIXEME：没有写完
		# 在 feature maps 上产生大约 2w 个 anchors
		anchors = generate_anchors(self._feature_maps_shape[0], self._feature_maps_shape[1])

		tf.logging.info('Totoally generates {} anchors.'.format(anchors.shape[0]))

		# `[H * W * 18]`  `[H * W * 36]`
		rpn_scores, rpn_coordinates = self._rpn_head_creator(feature_maps)

		# 其实就是 proposal layer
		# 主要是将经过 RPN Head 的 fg/bg 分类结果 和 经过 reg 的偏移值结果传入
		# 用于生成进入 RoI Pooling layer 的 proposals
		rpn_proposals = self._rpn_proposals_creator((rpn_scores, rpn_coordinates, anchors, image_shape), 
												training=training)

		if training:
			# 从 `20000` 多个候选的 anchors 选出 `256` 个 anchors 进行分类和回归位置。
			rpn_labels, rpn_target_anchors, rpn_w_in, rpn_w_out = \
				self._rpn_target_anchors_creator((gt_bboxes, image_shape, anchors), 
												training=training)
			
			# rpn loss
			# 其实就是进行 RPN 训练
			rpn_cls_loss, rpn_reg_loss = rpn_losses_creator(self._num_anchors, rpn_scores, rpn_coordinates,
															rpn_labels, rpn_target_anchors,
															rpn_w_in, rpn_w_out)
			
			# roi loss
			final_roi_proposals, final_roi_labels, final_roi_target_bboxes, roi_w_in, roi_w_out = \
						self._roi_target_proposals_creator((rpn_proposals, gt_bboxes, gt_labels),
															training)
			# feature_maps, proposals, spatial_scale = inputs
			roi_features = self._roi_pooling_creator((feature_maps, final_roi_proposals, self._feature_stride),
													training=training)
			roi_scores, roi_coordinates = self._roi_head_creator(roi_features, training=training)	

			roi_cls_loss, roi_reg_loss = roi_losses_creator(roi_scores, roi_coordinates, final_roi_labels,
															final_roi_target_bboxes, roi_w_in, roi_w_out)				
			return rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss

	def roi_head_creator(self):
		raise NotImplementedError
	
	def feature_extractor_creator(self):
		raise NotImplementedError


							


