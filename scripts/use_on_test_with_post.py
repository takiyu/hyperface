#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

import argparse
import cv2
import os
import multiprocessing
import numpy as np
import six
import time

import common
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

def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


def _forward_with_rects(model, img_org, rects, batchsize):
    # Crop and normalize
    cropped_imgs = list()
    for x, y, w, h in rects:
        img = img_org[int(y):int(y + h + 1), int(x):int(x + w + 1), :]
        img = cv2.resize(img, models.IMG_SIZE)
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        img = np.transpose(img, (2, 0, 1))
        cropped_imgs.append(img)

    detections = list()
    landmarks = list()
    visibilitys = list()
    poses = list()
    genders = list()

    # Forward each batch
    for i in six.moves.xrange(0, len(cropped_imgs), batchsize):
        # Create batch
        batch = xp.asarray(cropped_imgs[i:i+batchsize], dtype=np.float32)
        x = chainer.Variable(batch, volatile=True)
        # Forward
        y = model(x)
        # Chainer.Variable -> np.ndarray
        detections.extend(_cvt_variable(y['detection']))
        landmarks.extend(_cvt_variable(y['landmark']))
        visibilitys.extend(_cvt_variable(y['visibility']))
        poses.extend(_cvt_variable(y['pose']))
        genders.extend(_cvt_variable(y['gender']))

    # Denormalize landmarks
    for i, (x, y, w, h) in enumerate(rects):
        landmarks[i] = landmarks[i].reshape(models.N_LANDMARK, 2)  # (21, 2)
        landmark_offset = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        landmark_denom = np.array([w, h], dtype=np.float32)
        landmarks[i] = landmarks[i] * landmark_denom + landmark_offset

    return detections, landmarks, visibilitys, poses, genders


def _propose_region(prev_rect, pts, pad_rate=0.3):
    rect = cv2.boundingRect(pts)
    # padding
    x, y, w, h = rect
    pad_w = w * pad_rate / 2.0
    pad_h = h * pad_rate / 2.0
    rect = (x - pad_w, y - pad_h, w + pad_w, h + pad_h)
    # union
    x = max(rect[0], prev_rect[0])
    y = max(rect[1], prev_rect[1])
    w = min(rect[0] + rect[2], prev_rect[0] + prev_rect[2]) - x
    h = min(rect[1] + rect[3], prev_rect[1] + prev_rect[3]) - y
    return (x, y, w, h)


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
                                  config.aflw_test_rate, raw_mode=True)

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
        logger.info('Next image')

        # Load AFLW test
        entry = test[cnt_img]
        img = cv2.imread(entry['img_path'])
        img = img.astype(np.float32) / 255.0  # [0:1]

        # Iterative Region Proposals (IRP)
        detections, landmarks = None, None
        for stage_cnt in six.moves.xrange(2):
            if stage_cnt == 0:
                # Selective search, crop and normalize
                ssrects = common.selective_search_dlib(img, min_size=2500,
                                                       check=False,
                                                       debug_window=False)
            else:
                new_ssrects = list()
                for i in six.moves.xrange(len(ssrects)):
                    if detections[i] > config.detection_threshold:
                        new_ssrect = _propose_region(ssrects[i], landmarks[i])
                        new_ssrects.append(new_ssrect)
                ssrects = new_ssrects

            # Forward
            detections, landmarks, visibilitys, poses, genders = \
                    _forward_with_rects(model, img, ssrects, config.batchsize)

        # Draw all
        for i in six.moves.xrange(len(ssrects)):
            detection = detections[i]
            landmark = landmarks[i]
            visibility = visibilitys[i]
            pose = poses[i]
            gender = genders[i]

            detection = (detection > config.detection_threshold)

            # Draw results
            drawing.draw_detection(img, detection)
            landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
            drawing.draw_landmark(img, landmark, visibility, landmark_color,
                                  0.5, denormalize_scale=False)
            drawing.draw_pose(img, pose)
            drawing.draw_gender(img, gender)

        # Send to imgviewer
        img *= 255  # [0:1] -> [0:255]
        max_cnt = 66
        if detection:
            viewer_que.put(('○', '{}'.format(cnt_o), {'img': img}))
            cnt_o = (cnt_o + 1) % max_cnt
        else:
            viewer_que.put(('☓', '{}'.format(cnt_x), {'img': img}))
            cnt_x = (cnt_x + 1) % max_cnt
        cnt_img = (cnt_img + 1) % len(test)

        time.sleep(1.0)
