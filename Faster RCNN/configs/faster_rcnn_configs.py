def _get_faster_rcnn_pascalvoc_configs():
    return {
        # vgg16
        'roi_feature_size_vgg16': (7, 7, 512),
        'roi_head_keep_dropout_rate': 0.5,
        'roi_pooling_max_pooling_vgg16': True,

        # resnet
        'roi_feature_size_resnet': (7, 7, 1024),
        'roi_pooling_max_pooling_resnet': False,

        # training configs
        'learning_rate': 1e-4,
        'learning_rate_multi_decay_steps': [80000],
        'learning_rate_multi_lrs': [1e-3, 1e-4],
        'learning_rate_bias_double': True,
        'momentum': 0.9,
        'epoches': 8,
        'image_min_size': 600,
        'image_max_size': 1000,
        'preprocessing_type': 'caffe',
        'bgr_pixel_means': [103.939, 116.779, 123.68],
        
        # general configs
        'num_classes': 21,
        'weight_decay': 0.0001,
        'aspect_ratios': (0.5, 1, 2),
        'anchor_steps': (8, 16, 32),
        'feature_stride': 16,

        # rpn proposals 参数
        'rpn_proposals_nms_threshold': 0.7,
        'rpn_proposals_num_pre_nms_train': 12000,
        'rpn_proposals_num_post_nms_train': 2000,
        'rpn_proposals_num_pre_nms_test' : 6000,
        'rpn_proposals_num_post_nms_test': 300,

        # rpn target anchors 参数
        'rpn_target_anchors_positive_iou_threshold': 0.7,
        'rpn_target_anchors_negative_iou_threshold': 0.3,
		'rpn_target_anchors_total_samples': 256,
        'rpn_target_anchors_max_positive_samples': 128,
									
        # roi pooling 参数
        'roi_pooling_size': 7,
        'roi_pooling_max_pooling': True,

        'roi_training_positive_iou_threshold': .5,
        'roi_training_negative_iou_threshold': .1,
        'roi_training_total_num_samples': 128,
        'roi_training_max_positive_samples': 32,
    }


# def get_faster_rcnn_coco_hyperparams():


pasalvoc_configs = _get_faster_rcnn_pascalvoc_configs()

