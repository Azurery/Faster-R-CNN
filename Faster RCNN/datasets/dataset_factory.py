# from datasets.coco import get_training_dataset as get_coco_train_dataset
# from datasets.coco import get_eval_dataset as get_coco_eval_dataset

from datasets.pascalvoc import get_dataset as get_pascal_train_dataset
from datasets.pascalvoc import get_dataset_by_local_file as get_pascal_eval_dataset


def dataset_factory(dataset_class, mode, configs):
    if dataset_class == 'pascalvoc':
        if mode == 'trainval':
            return get_pascal_train_dataset(**configs)
        elif mode == 'test':
            return get_pascal_eval_dataset(**configs)
        return ValueError('unknown mode {} for dataset class type {}.'.format(mode, dataset_class))

    # if dataset_class == 'coco':
    #     if mode == 'train':
    #         return get_coco_train_dataset(**configs)
    #     elif mode == 'val':
    #         return get_coco_eval_dataset(**configs)
    #     raise ValueError('unknown mode {} for dataset class type {}.'.format(mode, dataset_class))
    
    # raise ValueError('unknown dataset type {}.'.format(dataset_class))