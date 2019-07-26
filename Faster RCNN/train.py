import os
import time
import re
import argparse
import tensorflow as tf
import numpy as np
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from tensorflow.contrib.summary import summary
from tensorflow.contrib.eager.python import saver as model_saver
from tqdm import tqdm

from utils.visual_utils import show_image
from configs.configs_factory import configs_factory
from model.model_factory import model_factory
from datasets.dataset_factory import dataset_factory

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser()
configs = None

# tf.flags.DEFINE_integer()

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
tf.logging.set_verbosity(tf.logging.INFO)

def train_one_epoch(dataset, base_model, optimizer, preprocessing_type, saver, 
                    save_path, summary_every_n_steps, save_every_n_steps, logging_every_n_steps):
    index = 0
    for image, gt_bboxes, gt_labels in tqdm(dataset):
        gt_bboxes = tf.squeeze(gt_bboxes, axis=0)
        coordinates = tf.split(gt_bboxes, 4, axis=1)
        gt_bboxes = tf.concat([coordinates[1], coordinates[0], coordinates[3], coordinates[2]], axis=1)

        gt_labels = tf.cast(tf.squeeze(gt_labels, axis=0), tf.int32)
        with tf.GradientTape() as tape:
            # image = tf.cast(image, dtype=tf.float32)
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = base_model((image, gt_bboxes, gt_labels), True)
            l2_loss = tf.add_n(base_model.losses)
            total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + l2_loss
            
            # 这部分不太懂
            all_variables = base_model.variables
            gradients = tape.gradient(total_loss, all_variables)

            if configs['learning_rate_bias_double']:
                all_gradients = []
                all_variables = []
                for gradient, variable in zip(gradients, base_model.variables):
                    if gradient is None:
                        continue
                    scale = 1.0
                    if 'bias' in variable.name:
                        scale = 2.0
                    all_gradients.append(gradient * scale)
                    all_variables.append(variable)
                gradients = all_gradients
            optimizer.apply_gradients(zip(gradients, all_variables), global_step=tf.train.get_or_create_global_step())

        # loss summary
        if index % summary_every_n_steps == 0:
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('rpn_cls_loss', rpn_cls_loss)
            tf.summary.scalar('rpn_reg_loss', rpn_reg_loss)
            tf.summary.scalar('roi_cls_loss', roi_cls_loss)
            tf.summary.scalar('roi_reg_loss', roi_reg_loss)
            tf.summary.scalar('total_loss', total_loss)

        # image summary
        # pred_bboxes, pred_lables, pred_scores = base_model(image, False)
        # if pred_bboxes is not None:
        #     pred_image_index = tf.where(pred_scores >= configs['show_image_score_threshold'])
        #     if tf.size(pred_image_index) != 0:
        #         # show ground truth
        #         gt_coordinates = tf.split(gt_bboxes, 4, axis=1)
        #         show_gt_bboxes = tf.concat([gt_coordinates[1], gt_coordinates[0], gt_coordinates[3], gt_coordinates[2]], axis=1)
        #         gt_image = show_image(tf.squeeze(image, axis=0).numpy(), show_gt_bboxes.numpy(), pred_lables.numpy,
        #                             preprocessing_type=preprocessing_type, caffe_pixel_means=configs['bgr_pixel_means'],
        #                             enable_matplotlib=False)
                
        #         tf.summary.image('gt_image', tf.expand_dims(gt_image, axis=0))

        #         # show pred
        #         pred_bboxes = tf.gather(pred_bboxes, pred_image_index)
        #         pred_labels = tf.gather(pred_labels, pred_image_index)
        #         pred_coordinates = tf.split(pred_bboxes, 4, axis=1)
        #         show_pred_bboxes = tf.concat([pred_coordinates[1], pred_coordinates[0], pred_coordinates[3], pred_coordinates[2]], axis=1)
        #         pred_image = show_image(tf.squeeze(image, axis=0).numpy(), show_pred_bboxes.numpy(), pred_labels.numpy(),
        #                                 preprocessing_type=preprocessing_type, caffe_pixel_means=configs['bgr_pixel_means'],
        #                                 enable_matplotlib=False)
        #         tf.summary.image('pred_image', tf.expand_dims(pred_image, axis=0))

        # logging
        if index % logging_every_n_steps == 0:
            if isinstance(optimizer, tf.train.AdamOptimizer):
                show_learning_rate = optimizer._lr()
            else:
                show_learning_rate = optimizer._learning_rate()
        
        tf.compat.v1.logging.info('steps %d, lr: %.5f\nloss: rpn_cls_loss: %.4f,  rpn_reg_loss: %.4f,  roi_cls_loss: %.4f,  roi_reg_loss: %.4f\ntotal_loss: %.4f' % 
                        (index + 1, show_learning_rate, rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss, total_loss))

        # save
        if saver is not None and save_path is not None and index % save_every_n_steps == 0:
            saver.save(os.path.join(save_path, 'model.ckpt'), global_step=tf.train.get_or_create_global_step())

        index += 1


def train(training_dataset, preprocessing_type, base_model, optimizer, summary_dir,
        ckpt_dir, logging_every_n_steps, summary_every_n_steps, save_every_n_steps,
        restore_ckpt_file_path):
    # 获取 pretained model
    variables = base_model.variables + [tf.train.get_or_create_global_step()]
    saver = model_saver.Saver(variables)

    if restore_ckpt_file_path is not None:
        saver.restore(restore_ckpt_file_path)
    
    if tf.train.latest_checkpoint(ckpt_dir) is not None:
        saver.restore(tf.train.latest_checkpoint(ckpt_dir))
    

    # writer = tf.summary.FileWriter(train_dir, flush_secs=100)
    writer = summary.create_file_writer(summary_dir, flush_millis=100000)
    for i in range(configs['epoches']):
        tf.compat.v1.logging.info('epoch %d starting...' % (i + 1))
        start_time = time.time()
        with writer.as_default(), summary.always_record_summaries():
            train_one_epoch(dataset=training_dataset, base_model=base_model, optimizer=optimizer, preprocessing_type=preprocessing_type,
                        logging_every_n_steps=logging_every_n_steps, summary_every_n_steps=summary_every_n_steps,
                        save_path=ckpt_dir, saver=saver, save_every_n_steps=save_every_n_steps)
        tf.set_random_seed(1)
        end_time = time.time()
        tf.compat.v1.logging.info('epoch %d training finished, costing %d seconds...' % (i, end_time - start_time))


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--gpu_id', default='0', type=str, help='used in sys variable CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--model_class', default='faster_rcnn', type=str, help='one of [faster_rcnn, fpn].')
    parser.add_argument('--backbone', default='resnet50', type=str, help='one of [vgg16, resnet50, resnet101, resnet152].')
    
    parser.add_argument('--data_root_path', default='tfrecords/', type=str)
    parser.add_argument('--dataset_class', default='pascalvoc', type=str, help='one of [pascalvoc, coco].')
    parser.add_argument('--coco_year', default='2017', type=str, help='one of [2014, 2017].')
    parser.add_argument('--pascalvoc_year', default='2007', type=str, help='one of [2007, 2012].')
    parser.add_argument('--pascalvoc_class', default='trainval', type=str, help='one of [trainval, test].')
    parser.add_argument('--num_pascalvoc_tfrecords', default=2, type=int, help='num of pascalvoc tfrecords.')
    parser.add_argument('--use_adam', type=bool, default=False)
    parser.add_argument('--logging_every_n_steps', default=100, type=int)
    parser.add_argument('--saving_every_n_steps', default=5000, type=int)
    parser.add_argument('--summary_every_n_steps', default=100, type=int)
    parser.add_argument('--restore_ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', default='checkpoints/', type=str)
    parser.add_argument('--restore_ckpt_file_path', type=str, default=None)
    parser.add_argument('--summary_dir', type=str, default='summary/')


    # parser.add_argument('log_')
    args = parser.parse_args()
    return args
        

def _get_dataset(tfrecords_dir='tfrecords/', preprocessing_type='caffe', dataset_class='pascalvoc', coco_year='2017',
                pascalvoc_year='2007', pascalvoc_class='trainval', num_pascalvoc_tfrecords=2,
                data_root_path=None):
    if dataset_class == 'pascalvoc':
        file_pattern = 'pascalvoc_{}_{}_%02d.tfrecord'.format(pascalvoc_year, pascalvoc_class)
        file_names = [os.path.join(tfrecords_dir, file_pattern % i) for i in range(num_pascalvoc_tfrecords)]
        # print(file_names)
        dataset_configs = { 'tf_record_list': file_names,
                            'min_size': configs['image_min_size'],
                            'max_size': configs['image_max_size'],
                            'preprocessing_type': configs['preprocessing_type'],
                            'caffe_pixel_means': configs['bgr_pixel_means'],
                            'data_argumentation': True
                            }
        # def dataset_factory(dataset_class, mode, configs):
        dataset = dataset_factory('pascalvoc', 'trainval', dataset_configs)
        return dataset
    # elif dataset_class == 'coco'

def _get_optimizer(use_adam):
    learning_rate = tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                                boundaries=configs['learning_rate_multi_decay_steps'],
                                                values=configs['learning_rate_multi_lrs'])
    if use_adam:
        return tf.train.AdamOptimizer(learning_rate)
    else:
        return tf.train.MomentumOptimizer(learning_rate, momentum=configs['momentum'])



def main(args):
    global configs
    # def hyperparams_factory(dataset_class, model_class):    
    configs = configs_factory(args.dataset_class, args.model_class)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    #　建立模型
    # def model_factory(model_class, backbone, configs):
    model = model_factory(args.model_class, args.backbone, configs)
    # model()
    model(tf.cast(np.random.rand(1, 800, 600, 3), dtype=tf.float32), False)

    # log基本信息
    # log_path_name = 'log_{}_{}_{}_{}'.format(args.dataset_class, args.model_class, args.log_name)

    # preprocessing_type = 'caffe'

    # 开始训练
    train(training_dataset=_get_dataset(preprocessing_type='caffe', dataset_class=args.dataset_class,
                                        coco_year=args.coco_year, pascalvoc_year=args.pascalvoc_year,
                                        pascalvoc_class=args.pascalvoc_class, num_pascalvoc_tfrecords=args.num_pascalvoc_tfrecords,
                                        data_root_path=args.data_root_path),
        preprocessing_type='caffe',
        base_model=model,
        optimizer=_get_optimizer(args.use_adam),
        logging_every_n_steps=args.logging_every_n_steps,
        save_every_n_steps=args.saving_every_n_steps,
        summary_every_n_steps=args.summary_every_n_steps,
        summary_dir=args.summary_dir,
        ckpt_dir=args.ckpt_dir,
        restore_ckpt_file_path=args.restore_ckpt_file_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)