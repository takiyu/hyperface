# -*- coding: utf-8 -*-
import cv2
import dlib
import math

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _scale_down_image(img, max_img_size):
    org_h, org_w = img.shape[0:2]
    h, w = img.shape[0:2]
    if max_img_size[0] < w:
        h *= float(max_img_size[0]) / float(w)
        w = max_img_size[0]
    if max_img_size[1] < h:
        w *= float(max_img_size[1]) / float(h)
        h = max_img_size[1]
    # Apply resizing
    if h == org_h and w == org_w:
        resize_scale = 1
    else:
        resize_scale = float(org_h) / float(h)  # equal to `org_w / w`
        img = cv2.resize(img, (int(w), int(h)))
    return img, resize_scale


def selective_search_dlib(img, max_img_size=(500, 500),
                          kvals=(50, 200, 2), min_size=2200, check=True,
                          debug_window=False):
    if debug_window:
        org_img = img
    org_h, org_w = img.shape[0:2]

    # Resize the image for speed up
    img, resize_scale = _scale_down_image(img, max_img_size)

    # Selective search
    drects = []
    dlib.find_candidate_object_locations(img, drects, kvals=kvals,
                                         min_size=min_size)
    rects = [(int(drect.left() * resize_scale),
              int(drect.top() * resize_scale),
              int(drect.width() * resize_scale),
              int(drect.height() * resize_scale)) for drect in drects]

    # Check the validness of the rectangles
    if check:
        if len(rects) == 0:
            logger.error('No selective search rectangle '
                         '(Please tune the parameters)')
        for rect in rects:
            x, y = rect[0], rect[1]
            w, h = rect[2], rect[3]
            x2, y2 = x + w, y + h
            if x < 0 or y < 0 or org_w < x2 or org_h < y2 or w <= 0 or h <= 0:
                logger.error('Invalid selective search rectangle, rect:{}, '
                             'image:{}'.format(rect, (org_h, org_w)))

    # Debug window
    if debug_window:
        for rect in rects:
            p1 = (rect[0], rect[1])
            p2 = (rect[0] + rect[2], rect[1] + rect[3])
            cv2.rectangle(org_img, p1, p2, (0, 255, 0))
        cv2.imshow('selective_search_dlib', org_img)
        cv2.waitKey(0)

    return rects


def rect_or(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def rect_and(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return (0, 0, 0, 0)
    return (x, y, w, h)


def rect_area(a):
    return a[2] * a[3]


def rect_overlap_rate(a, b):
    area_and = rect_area(rect_and(a, b))
    area_or = rect_area(rect_or(a, b))
    if area_or == 0:
        return 0
    else:
        return math.sqrt(float(area_and) / float(area_or))
