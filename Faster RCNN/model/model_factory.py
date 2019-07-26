from nets.faster_rcnn.faster_rcnn_vgg16 import faster_rcnn_vgg16
from nets.faster_rcnn.faster_rcnn_resnet import faster_rcnn_resnet


def model_factory(model_class, backbone, configs):
    if model_class == 'faster_rcnn':
        if backbone == 'vgg16':
            return _get_faster_rcnn_vgg16(None, configs)
        elif backbone == 'resnet50':
            return _get_faster_rcnn_resnet(50, configs)
        elif backbone == 'resnet101':
            return _get_faster_rcnn_resnet(101, configs)
        elif backbone == 'resnet152':
            return _get_faster_rcnn_resnet(152, configs)
        else:
            raise ValueError('unknown backbone {}.'.format(backbone))
    # # elif model_class == 'fpn':
    #     return _get_fpn_resnet(configs[depth])

def _get_faster_rcnn_vgg16(slim_ckpt_file_path, configs):
    return faster_rcnn_vgg16(slim_ckpt_file_path=slim_ckpt_file_path,
                            roi_head_keep_dropout_rate=configs['roi_head_keep_dropout_rate'],
                            roi_feature_size=configs['roi_feature_size_vgg16'],
                            num_classes=configs['num_classes'],
                            weight_decay=configs['weight_decay'],
                            anchor_steps=configs['anchor_steps'],
                            aspect_ratios=configs['aspect_ratios'],
                            rpn_proposals_nms_threshold=configs['rpn_proposals_nms_threshold'],
                            rpn_proposals_num_pre_nms_train=configs['rpn_proposals_num_pre_nms_train'],
                            rpn_proposals_num_post_nms_train=configs['rpn_proposals_num_post_nms_train'],
                            rpn_proposals_num_pre_nms_test=configs['rpn_proposals_num_pre_nms_test'],
                            rpn_proposals_num_post_nms_test=configs['rpn_proposals_num_post_nms_test'],

                            # rpn target anchors 参数
                            rpn_target_anchors_positive_iou_threshold=configs['rpn_target_anchors_positive_iou_threshold'],
                            rpn_target_anchors_negative_iou_threshold=configs['rpn_target_anchors_negative_iou_threshold'],
                            rpn_target_anchors_total_samples=configs['rpn_target_anchors_total_samples'],
                            rpn_target_anchors_max_positive_samples=configs['rpn_target_anchors_max_positive_samples'],

                            # roi pooling 参数
                            roi_pooling_size=configs['roi_pooling_size'],
                            roi_pooling_max_pooling=configs['roi_pooling_max_pooling'],

                            roi_training_positive_iou_threshold=configs['roi_training_positive_iou_threshold'],
                            roi_training_negative_iou_threshold=configs['roi_training_negative_iou_threshold'],
                            roi_training_total_num_samples=configs['roi_training_total_num_samples'],
                            roi_training_max_positive_samples=configs['roi_training_max_positive_samples'],)

def _get_faster_rcnn_resnet(depth, configs):
    return faster_rcnn_resnet(depth=depth,
                            roi_head_keep_dropout_rate=configs['roi_head_keep_dropout_rate'],
                            roi_feature_size=configs['roi_feature_size_resnet'],
                            num_classes=configs['num_classes'],
                            weight_decay=configs['weight_decay'],
                            anchor_steps=configs['anchor_steps'],
                            aspect_ratios=configs['aspect_ratios'],
                            rpn_proposals_nms_threshold=configs['rpn_proposals_nms_threshold'],
                            rpn_proposals_num_pre_nms_train=configs['rpn_proposals_num_pre_nms_train'],
                            rpn_proposals_num_post_nms_train=configs['rpn_proposals_num_post_nms_train'],
                            rpn_proposals_num_pre_nms_test=configs['rpn_proposals_num_pre_nms_test'],
                            rpn_proposals_num_post_nms_test=configs['rpn_proposals_num_post_nms_test'],

                            # rpn target anchors 参数
                            rpn_target_anchors_positive_iou_threshold=configs['rpn_target_anchors_positive_iou_threshold'],
                            rpn_target_anchors_negative_iou_threshold=configs['rpn_target_anchors_negative_iou_threshold'],
                            rpn_target_anchors_total_samples=configs['rpn_target_anchors_total_samples'],
                            rpn_target_anchors_max_positive_samples=configs['rpn_target_anchors_max_positive_samples'],

                            # roi pooling 参数
                            roi_pooling_size=configs['roi_pooling_size'],
                            roi_pooling_max_pooling=configs['roi_pooling_max_pooling_resnet'],

                            roi_training_positive_iou_threshold=configs['roi_training_positive_iou_threshold'],
                            roi_training_negative_iou_threshold=configs['roi_training_negative_iou_threshold'],
                            roi_training_total_num_samples=configs['roi_training_total_num_samples'],
                            roi_training_max_positive_samples=configs['roi_training_max_positive_samples'])

# def _get_fpn_resnet(depth, configs):
#     return fpn_resnet()