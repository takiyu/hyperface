# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np

from chainer import variable
from chainer.training import extension

from . import imgviewer

# logging
from logging import getLogger, NullHandler, INFO
logger = getLogger(__name__)
logger.addHandler(NullHandler())
# logging for imgviewer
imgviewer.logger.setLevel(INFO)

default_img_key = 'img'
default_batch_idx = 0


def _default_image_func(key, entry, viewer):
    # Get image in the entry
    img = entry.pop(default_img_key)
    # Batch extraction
    if img.ndim == 4:
        img = img[default_batch_idx]
    # Transpose image channels
    if img.ndim == 3 and (img.shape[0] == 1 or img.shape[0] == 3):
        img = np.transpose(img, (1, 2, 0))
        img = img.copy()  # for safety
    return {'img': img}


def _default_entry_func(key, viewer, trainer):
    if key in trainer.observation:
        return trainer.observation[key]
    else:
        return None


def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


class ImgViewerExtention(extension.Extension):

    def __init__(self, keys, n_imgs, port=5000, entry_func=None,
                 image_func=None):
        assert(len(keys) == len(n_imgs))
        img_cnt_max = {k: n for k, n in zip(keys, n_imgs)}
        img_cnts = {k: 0 for k in keys}

        # Start image viewer
        viewer_que = multiprocessing.Queue()
        imgviewer.start(viewer_que, stop_page=False, port=port)

        # Member variables
        self._keys = keys
        self.img_cnt_max = img_cnt_max
        self.img_cnts = img_cnts
        self._viewer_que = viewer_que
        self._entry_func = entry_func
        self._image_func = image_func

    def __call__(self, trainer):
        for key in self._keys:
            # Create an entry
            if self._entry_func:
                entry = self._entry_func(key, self, trainer)  # User function
            else:
                entry = _default_entry_func(key, self, trainer)

            # Skip the current key
            if entry is None:
                continue

            # Convert to numpy.ndarray
            entry = {k: _cvt_variable(v) for k, v in entry.items()}

            # Create image from an entry(dict)
            if self._image_func:
                data = self._image_func(key, entry, self)  # User function
            else:
                data = _default_image_func(key, entry, self)

            # Send images
            if not isinstance(data, list):
                data = [data]
            for v in data:
                name = 'cnt_{}'.format(self.img_cnts[key])
                self.img_cnts[key] = \
                    (self.img_cnts[key] + 1) % self.img_cnt_max[key]
                self._viewer_que.put((key, name, v))
