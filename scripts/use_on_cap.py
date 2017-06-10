#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

import argparse
import cv2
import datasets
import os
import numpy as np

import config
import drawing
import log_initializer
import models

# logging
from logging import getLogger, DEBUG
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"

def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error('Failed to open camera')
        exit()

    # Main loop
    logger.info('Start main loop')
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.resize(img, datasets.IMG_SIZE)
        img = img.astype(np.float32)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))

        # Create single batch
        imgs = xp.asarray([img])
        x = chainer.Variable(imgs, volatile=True)

        # Forward
        y = model(x)

        # Chainer.Variable -> np.ndarray
        imgs = _cvt_variable(y['img'])
        detections = _cvt_variable(y['detection'])
        landmarks = _cvt_variable(y['landmark'])
        visibilitys = _cvt_variable(y['visibility'])
        poses = _cvt_variable(y['pose'])
        genders = _cvt_variable(y['gender'])

        # Use first data in one batch
        img = imgs[0]
        detection = detections[0]
        landmark = landmarks[0]
        visibility = visibilitys[0]
        pose = poses[0]
        gender = genders[0]

        img = np.transpose(img, (1, 2, 0))
        img = img.copy()
        img += 0.5  # [-0.5:0.5] -> [0:1]
        detection = (detection > 0.5)
        gender = (gender > 0.5)

        # Draw results
        drawing.draw_detection(img, detection)
        landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
        drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5)
        drawing.draw_pose(img, pose)
        drawing.draw_gender(img, gender)

        cv2.imshow('img', img)
        cv2.waitKey(1)
