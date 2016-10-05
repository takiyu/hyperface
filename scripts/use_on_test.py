#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

import argparse
import os
import multiprocessing
import numpy as np
import time

import config
import datasets
import drawing
import log_initializer
import models
from extensions import imgviewer

# logging
from logging import getLogger, DEBUG, INFO
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)
# logging for imgviewer
imgviewer.logger.setLevel(INFO)

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='HyperFace training script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    parser.add_argument('--model', required=True, help='Trained model path')
    args = parser.parse_args()

    logger.info('HyperFace Evaluation')

    # Load config
    config.load(args.config)

    # Setup AFLW dataset
    _, test = datasets.setup_aflw(config.aflw_cache_path,
                                  config.aflw_sqlite_path,
                                  config.aflw_imgdir_path,
                                  config.aflw_test_rate)

    # Define a model
    logger.info('Define a HyperFace model')
    model = models.HyperFaceModel()
    model.train = False
    model.report = False
    model.backward = False

    # Initialize model
    logger.info('Initialize a model using model "{}"'.format(args.model))
    chainer.serializers.load_npz(args.model, model)

    # Setup GPU
    if config.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    # Start ImgViewer
    viewer_que = multiprocessing.Queue()
    imgviewer.start(viewer_que, stop_page=True, port=config.port_evaluate)

    # Main loop
    logger.info('Start main loop')
    cnt_img, cnt_o, cnt_x = 0, 0, 0
    while True:
        data = test[cnt_img]

        # Create single batch
        img = data['x_img']
        imgs = xp.asarray([img])
        x = chainer.Variable(imgs, volatile=True)

        # Forward
        y = model(x)

        # Chainer.Variable -> xp.ndarray
        img = y['img'].data
        detection = y['detection'].data
        landmark = y['landmark'].data
        visibility = y['visibility'].data
        pose = y['pose'].data
        gender = y['gender'].data

        # xp.ndarray -> np.ndarray
        if config.gpu >= 0:
            img = img.get()
            detection = detection.get()
            landmark = landmark.get()
            visibility = visibility.get()
            pose = pose.get()
            gender = gender.get()

        # Use first data in one batch
        img = img[0]
        detection = detection[0]
        landmark = landmark[0]
        visibility = visibility[0]
        pose = pose[0]
        gender = gender[0]

        img = np.transpose(img, (1, 2, 0))
        img = img.copy()
        img += 0.5  # [-0.5:0.5] -> [0:1]

        # Draw results
        drawing.draw_detection(img, detection)
        landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
        drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5)
        drawing.draw_pose(img, pose)
        drawing.draw_gender(img, gender)

        img *= 255  # [0:1] -> [0:255]

        max_cnt = 66
        if detection == 1:
            viewer_que.put(('○', '{}'.format(cnt_o), {'img': img}))
            cnt_o = (cnt_o + 1) % max_cnt
        else:
            viewer_que.put(('☓', '{}'.format(cnt_x), {'img': img}))
            cnt_x = (cnt_x + 1) % max_cnt
        cnt_img = (cnt_img + 1) % len(test)
        time.sleep(1.0)
