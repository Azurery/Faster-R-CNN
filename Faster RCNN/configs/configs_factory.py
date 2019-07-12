def configs_factory(dataset_class, model_class):
    if model_class == 'faster_rcnn':
        if dataset_class == 'pascalvoc':
            from configs.faster_rcnn_configs import pasalvoc_configs
            return pasalvoc_configs
        # elif dataset_class == 'coco':
            