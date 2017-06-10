#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import multiprocessing
import numpy as np
import six

import config
import drawing
from hyperface import HyperFace
import log_initializer
from extensions import imgliveuploader

# logging
from logging import getLogger, DEBUG, INFO
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)
# logging for imgliveuploader
imgliveuploader.logger.setLevel(INFO)

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

    # HyperFace Model
    hyperface = HyperFace(args.model, config.gpu, config.batchsize)

    # Start ImgLiveUploader
    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    imgliveuploader.start(request_queue, response_queue, stop_page=False,
                          port=config.port_demo)

    # Main loop
    logger.info('Start main loop')
    while True:
        try:
            # Wait for image
            img = request_queue.get(timeout=3.0)
            img = img.astype(np.float32) / 255.0  # [0:1]

            logger.info('Next image')

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

            # Return to client
            img *= 255  # [0:1] -> [0:255]
            response_queue.put({'img': img}, timeout=1.0)

        except:
            pass
