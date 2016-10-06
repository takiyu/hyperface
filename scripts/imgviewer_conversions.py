# -*- coding: utf-8 -*-
'''
 Conversion functions for image viewer extension
'''

import cv2
import six
import numpy as np

import drawing

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def face_img_func(key, entry, viewer):
    # Image conversion
    img = entry['img'][0]   # Use only a first data in the batch
    assert(img.ndim == 3 and (img.shape[0] == 1 or img.shape[0] == 3))
    img = np.transpose(img, (1, 2, 0))
    img = img.copy()  # for safety
    img += 0.5  # [-0.5:0.5] -> [0:1]

    # Draw
    try:
        detection_raw = entry['detection'][0]
        detection = (detection_raw > 0.5)
        if 0.0 <= detection_raw <= 1.0:
            drawing.draw_detection(img, detection)

        landmark = entry['landmark'][0]
        visibility = entry['visibility'][0]
        landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
        drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5)

        pose = entry['pose'][0]
        drawing.draw_pose(img, pose)

        gender = entry['gender'][0]
        if 0.0 <= gender <= 1.0:
            gender = (gender > 0.5)
            drawing.draw_gender(img, gender)

    except KeyError:
        pass

    img = (img * 255).astype(np.uint8)
    caption = '{:02d}'.format(viewer.img_cnts[key])
    return {'img': img, 'cap': caption}


def weights_img_func(key, entry, viewer):
    data = entry['weights']
    assert(data.ndim == 4)
    img_cnt_max = viewer.img_cnt_max[key]

    res_data = list()

    # accumulate to 3 channels image
    for i in six.moves.range(min(data.shape[0], img_cnt_max)):
        img_shape = (3,) + data.shape[2:4]
        accum = np.zeros(img_shape, dtype=data.dtype)
        for ch in six.moves.range(data.shape[1]):
            accum[ch % 3] += data[i][ch]

        # normalize
        img = np.transpose(accum, (1, 2, 0))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        width = img.shape[0] * 15
        res_data.append({'img': img, 'width': width})

    return res_data


# ========================= Loss Graph (In a tab page) ========================
def lossgraph_entry_func(key, viewer, trainer):
    # Get a log
    log_report = trainer.get_extension('LogReport')
    log = log_report.log

    # Convert log to lists
    def extract_log(log, key, epoch_key):
        loss, epoch = list(), list()
        # TODO Consider duplication of epoch numbers
        for i, row in enumerate(log):
            if key in row and epoch_key in row:
                loss.append(row[key])
                epoch.append(row[epoch_key])
        return loss, epoch

    # Create a graph image from log
    def create_graph_img(log, kind):
        train_key = 'main/{}'.format(kind)
        test_key = 'validation/main/{}'.format(kind)
        train_loss, train_epoch = extract_log(log, train_key, 'epoch')
        test_loss, test_epoch = extract_log(log, test_key, 'epoch')
        if len(train_loss) == 0 and len(test_loss) == 0:
            return None
        else:
            return drawing.draw_loss_graph(train_loss, test_loss,
                                           train_epoch, test_epoch, title=kind)

    # Create loss graphs
    res = dict()
    loss_kinds = ['loss', 'loss_detection', 'loss_landmark', 'loss_visibility',
                  'loss_pose', 'loss_gender']
    for k in loss_kinds:
        img = create_graph_img(log, k)
        if img is not None:  # Use only valid ones
            res[k] = img
    return res


def lossgraph_img_func(key, entry, viewer):
    # Convert to viewer format
    return [{'img': entry[k]} for k in entry.keys()]
