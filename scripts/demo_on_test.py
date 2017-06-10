#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import multiprocessing
import numpy as np
import six
import time

import config
import datasets
import drawing
from hyperface import HyperFace
import log_initializer
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
                                  config.aflw_test_rate, raw_mode=True)

    # HyperFace Model
    hyperface = HyperFace(args.model, config.gpu, config.batchsize)

    # Start ImgViewer
    viewer_que = multiprocessing.Queue()
    imgviewer.start(viewer_que, stop_page=False, port=config.port_demo)

    # Main loop
    logger.info('Start main loop')
    cnt_img, cnt_view = 0, 0
    while True:
        logger.info('Next image (Count: {})'.format(cnt_img))

        # Load AFLW test
        entry = test[cnt_img]
        img = cv2.imread(entry['img_path'])
        img = img.astype(np.float32) / 255.0  # [0:1]

        # Resize image for the visibility
        img_height = 500
        img_width = img_height * img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_width), int(img_height)))

        # HyperFace
        landmarks, visibilities, poses, genders, rects = hyperface(img)

        # Draw results
        for i in six.moves.xrange(len(landmarks)):
            landmark = landmarks[i]
            visibility = visibilities[i]
            pose = poses[i]
            gender = genders[i]
            rect = rects[i]

            landmark_color = (0, 1, 0)  # detected color
            drawing.draw_landmark(img, landmark, visibility, landmark_color,
                                  0.5, denormalize_scale=False)
            drawing.draw_pose(img, pose, idx=i)
            gender = (gender > 0.5)
            # drawing.draw_gender(img, gender, idx=i)
            drawing.draw_gender_rect(img, gender, rect)

        # Send to imgviewer
        img *= 255  # [0:1] -> [0:255]
        max_cnt = 10
        viewer_que.put(('imgs', '{}'.format(cnt_view), {'img': img}))
        cnt_view = (cnt_view + 1) % max_cnt
        cnt_img = (cnt_img + 1) % len(test)

        time.sleep(1.0)
