"""
Define model in Keras and train with TensorFlow Estimator API.
Create the Estimator using model_to_estimator().
"""
import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import metrics
from camvid_dataset import random_crop
from spacenet_dataset import dataset
from keras_fc_densenet import build_FC_DenseNet
from metrics import sparse_categorical_accuracy, mean_iou
from common import ts_rand, get_available_cpus, delete_dir, pretty_tf_logging, backup_tf_checkpoint

tf.logging.set_verbosity(tf.logging.INFO)
    
def main(train_path, test_path, model_path, checkpoint_path, nb_epochs, batch_size, 
         image_height, image_width, crop_images, nb_crops, nb_classes, model_version):

    if checkpoint_path == None:
        model_dir = os.path.join(model_path, ts_rand())
        tf.logging.info('Create new checkpoint directory: %s' % model_path)
        os.makedirs(model_dir)
    else:
        tf.logging.info('Continue training from checkpoint directory: %s' % checkpoint_path)
        model_dir = checkpoint_path

    nb_cores = len(get_available_cpus())

    def train_input_fn():
        ds = dataset(train_path, num_parallel_reads=nb_cores)
        if crop_images:
            tf.logging.info('Create %d crops of size %dx%d from each image' % (nb_crops, image_height, image_width))
            ds = ds.flat_map(random_crop(nb_crops=nb_crops, random_flip=True, 
                             crop_height=image_height, crop_width=image_width))
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=5000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=100)
        return ds

    def eval_input_fn():
        ds = dataset(test_path, num_parallel_reads=nb_cores)
        if crop_images:
            # If the train images are cropped we need to crop the eval 
            # images as well so that they have the same input shape. 
            # We fix the seed to create a stable eval dataset.
            ds = ds.flat_map(random_crop(nb_crops=1, random_flip=False, 
                             crop_height=image_height, crop_width=image_width,
                             seed=0))
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=100)
        return ds

    fc_dn_model = build_FC_DenseNet(model_version=model_version, nb_classes=nb_classes, final_softmax=True, 
                                    input_shape=(image_height, image_width, 3), dropout_rate=0.2, 
                                    data_format='channels_last')

    #RMSprop(lr=0.001)
    fc_dn_model.compile(optimizer=Adam(lr=0.0001),
                        loss='sparse_categorical_crossentropy',
                        metrics=[sparse_categorical_accuracy, mean_iou(num_classes=nb_classes)])

    accuracy = tf.get_default_graph().get_tensor_by_name('metrics/sparse_categorical_accuracy/Mean:0')

    run_config = tf.estimator.RunConfig(save_summary_steps=50, 
                                        keep_checkpoint_max=100)

    # model_to_estimator creates a model_fn to initialize an Estimator. The 
    # EstimatorSpecs created by model_fn are initialized like this:
    #  - loss: will be set from model.total_loss
    #  - eval_metric_ops: will be set from model.metrics and model.metrics_tensors
    #  - predictions: will be set from model.output_names and model.outputs
    estimator = tf.keras.estimator.model_to_estimator(keras_model=fc_dn_model, 
                                                      model_dir=model_dir,
                                                      config=run_config)

    logged_tensors = {
        'global_step': tf.GraphKeys.GLOBAL_STEP, 
        'loss': fc_dn_model.total_loss.name,
        'accuracy': accuracy.name
        }

    best_loss = None
    best_loss_step = None
    for epoch in range(nb_epochs):
        tf.logging.info('Starting epoch %d' % epoch)
        train_hooks = [tf.train.LoggingTensorHook(tensors=logged_tensors, every_n_iter=1)]
        estimator.train(input_fn=train_input_fn, hooks=train_hooks)

        eval_hooks = [tf.train.LoggingTensorHook(tensors=logged_tensors, every_n_iter=1)]
        eval_results = estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)
        tf.logging.info('Epoch %d evaluation result: %s' % (epoch, eval_results))

        if best_loss == None or eval_results['loss'] < best_loss:
            best_loss = eval_results['loss']
            best_loss_step = eval_results['global_step']
            best_model_dir = os.path.join(model_dir, 'best_model')
            backup_tf_checkpoint(model_dir, best_loss_step, best_model_dir)
        tf.logging.info('best_loss: %f at step: %d' % (best_loss, best_loss_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', help='train data path', default='./camvid-preprocessed/camvid-384x480-train.tfrecords')
    parser.add_argument('--test-path', help='test data path', default='./camvid-preprocessed/camvid-384x480-test.tfrecords')
    parser.add_argument('--model-path', help='base directory for new checkpoints', default='./models')
    parser.add_argument('--checkpoint-path', help='directory for an existing checkpoint')
    parser.add_argument('--model-version', help='model version, one of [fcdn56,fcdn67,fcdn103]', default='fcdn56')
    parser.add_argument('--num-classes', help='number of classes', type=int, default=32)
    parser.add_argument('--num-epochs', help='number of epochs to train', type=int, default=100)
    parser.add_argument('--batch-size', help='batch size', type=int, default=5)
    parser.add_argument('--image-height', help='model input image height', type=int, default=224)
    parser.add_argument('--image-width', help='model input image width', type=int, default=224)
    parser.add_argument('--crop-images', help='crop and flip images randomly', action='store_true')
    parser.add_argument('--num-crops', help='number of crops per image, if cropping is enabled', type=int, default=5)
    args = parser.parse_args()
    main(train_path=args.train_path, test_path=args.train_path, model_path=args.model_path, 
         checkpoint_path=args.checkpoint_path, nb_epochs=args.num_epochs, batch_size=args.batch_size,  
         image_height=args.image_height, image_width=args.image_width, crop_images=args.crop_images, 
         nb_crops=args.num_crops, nb_classes=args.num_classes, model_version=args.model_version)
