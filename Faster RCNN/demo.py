import tensorflow as tf
import numpy as np
import tqdm
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

from utils.bbox_utils import generate_anchors, decode_bboxes, clip_bboxes
from utils.proposals_utils import rpn_head_creator
from nets.faster_rcnn_resnet import feature_extractor_resnet_creator
from nets.faster_rcnn_vgg16 import feature_extractor_creator_vgg16


if __name__ == '__main__':
    gt_bboxes = [[800., 4.8, 800., 331.19998],
                [488., 4.8, 800., 14.400001]]
    anchors = generate_anchors(38, 50, base_anchor_size=16, feature_stride=16, anchor_ratios=[0.5, 1, 2], anchor_steps=(8, 16, 32))
    extractor = feature_extractor_creator_vgg16()
    feature_maps = extractor()
    print(feature_maps.get_shape().as_list()[1:3])

    # rpn_head = rpn_head_creator(9, 0.001)
    # rpn_scores, rpn_coordinates = rpn_head(feature_maps)

    # decoded_bboxes = decode_bboxes(anchors, rpn_coordinates)

	# # 对选中且解码后的 anchors 进行 clip 和 filter
    # decoded_bboxes = clip_bboxes(decoded_bboxes, 600, 800)

    # num_nms_bbox = 2000
    # selected_indices = tf.image.non_max_suppression(tf.cast(decoded_bboxes, dtype=tf.float32),
	# 													rpn_scores,
	# 													max_output_size=2000,
	# 													iou_threshold=0.7)
    # selected_bboxes = tf.gather(decoded_bboxes, selected_indices)
    # print('rpn net generates %d proposals' % tf.size(selected_indices))
