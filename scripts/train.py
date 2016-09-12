#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

import argparse
import os

import config
import datasets
from extensions import ImgViewerExtention
from extensions import SequentialEvaluator
from imgviewer_conversions import face_img_func, weights_img_func
from imgviewer_conversions import lossgraph_entry_func, lossgraph_img_func
import log_initializer
import models
from models import copy_layers

# logging
from logging import getLogger, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='HyperFace training script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--pretrain', action='store_true',
                        help='Flag for pretraining mode')
    parser.add_argument('--pretrained_model', default='',
                        help='Initialize the model using pretrained')
    args = parser.parse_args()

    logger.info('HyperFace Training (Mode: {})'
                .format('Pretrain' if args.pretrain else 'Main'))

    # Load config
    config.load(args.config)

    # Setup AFLW dataset
    train, test = datasets.setup_aflw(config.aflw_cache_path,
                                      config.aflw_sqlite_path,
                                      config.aflw_imgdir_path,
                                      config.aflw_test_rate)

    # Define a model
    if args.pretrain:
        logger.info('Define a R-CNN_Face model')
        model = models.RCNNFaceModel()
        # Initialize using caffemodel
        if not args.resume:
            logger.info('Initialize a model using caffemodel "{}"'
                        .format(config.alexnet_caffemodel_path))
            caffe_model = CaffeFunction(config.alexnet_caffemodel_path)
            copy_layers(caffe_model, model)
    else:
        logger.info('Define a HyperFace model')
        model = models.HyperFaceModel()
        # Initialize using pretrained model
        if not args.resume and args.pretrained_model:
            logger.info('Initialize a model using pretraind "{}"'
                        .format(args.pretrained_model))
            pre_model = models.RCNNFaceModel()
            chainer.serializers.load_npz(args.pretrained_model, pre_model)
            copy_layers(pre_model, model)

    # Setup an optimizer
    logger.info('Setup an optimizer')
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Setup iterators
    # (test_iter needs repeat because of SequentialEvaluator())
    logger.info('Setup train/test iterators (n_loaders: {}/{})'
                .format(config.n_loaders_train, config.n_loaders_test))
    train_iter = iterators.MultiprocessIterator(train, config.batchsize, True,
                                                True, config.n_loaders_train)
    test_iter = iterators.MultiprocessIterator(test, config.batchsize, True,
                                               True, config.n_loaders_test)

    # Setup a updater
    logger.info('Setup an updater (GPU: {})'.format(config.gpus))
    updater = training.ParallelUpdater(train_iter, optimizer,
                                       devices=config.gpus)
    # Setup a trainer
    outdir = config.outdir_pretrain if args.pretrain else config.outdir
    logger.info('Setup a trainer (output directory: {})'.format(outdir))
    trainer = training.Trainer(updater, (config.n_epoch, 'epoch'), out=outdir)

    # Evaluation model with shared parameters
    logger.info('Create a copy of the model for evaluation')
    eval_model = model.copy()
    eval_model.train = False

    # Extension intervals
    n_iteration = max(len(train) / config.batchsize, 1)
    test_interval = (max(len(train) / len(test), 1), 'iteration')
    save_interval = (5, 'epoch')
    log_interval = (n_iteration / 1, 'iteration')
    imgview_face_interval = (10, 'iteration')
    imgview_weight_interval = (1, 'epoch')
    logger.info('Test interval : {}'.format(test_interval))
    logger.info('Save interval : {}'.format(save_interval))
    logger.info('Log interval :  {}'.format(log_interval))
    logger.info('ImgView face interval :   {}'.format(imgview_face_interval))
    logger.info('ImgView weight interval : {}'.format(imgview_weight_interval))

    # Extensions
    trainer.extend(extensions.dump_graph('main/loss'), trigger=save_interval)
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'), trigger=save_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'), trigger=save_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']),
        trigger=log_interval, invoke_before_training=True)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # My extensions
    # Sequential Evaluator
    trainer.extend(
        SequentialEvaluator(test_iter, eval_model, device=config.gpus['main']),
        trigger=test_interval)  # Sequential evaluation for imgviewer in test
    # Image Viewer for Faces
    trainer.extend(ImgViewerExtention(
        ['main/predict', 'main/teacher', 'validation/main/predict',
         'validation/main/teacher'], n_imgs=[20, 20, 10, 10],
        port=config.port_face, image_func=face_img_func),
        trigger=imgview_face_interval)
    # Image Viewer for weights
    trainer.extend(ImgViewerExtention(
        ['main/conv1_w', 'main/conv2_w', 'main/conv3_w', 'main/conv4_w',
         'main/conv5_w', ], n_imgs=[96, 256 / 2, 384 / 2, 384 / 2, 256 / 2],
        port=config.port_weight, image_func=weights_img_func),
        trigger=imgview_weight_interval)
    # Image Viewer for loss graph
    trainer.extend(ImgViewerExtention(
        ['lossgraph', ], n_imgs=[1], port=config.port_lossgraph,
        entry_func=lossgraph_entry_func, image_func=lossgraph_img_func),
        trigger=log_interval, invoke_before_training=True)

    # Resume
    if args.resume:
        logger.info('Resume from "{}"'.format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    logger.info('Start training')
    trainer.run()
