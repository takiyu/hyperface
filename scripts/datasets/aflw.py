# -*- coding: utf-8 -*-
import cv2
import dlib
import math
import numpy as np
import os.path
import random
import six
import sqlite3

import chainer

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Constant variables
N_LANDMARK = 21
IMG_SIZE = (227, 227)

# Python 2 compatibility
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def _exec_sqlite_query(cursor, select_str, from_str=None, where_str=None):
    query_str = 'SELECT {}'.format(select_str)
    query_str += ' FROM {}'.format(from_str)
    if where_str:
        query_str += ' WHERE {}'.format(where_str)
    return [row for row in cursor.execute(query_str)]


def _load_raw_aflw(sqlite_path, image_dir):
    ''' Load raw AFLW dataset from sqlite file
    Return:
        [dict('face_id', 'img_path', 'rect', 'landmark', 'landmark_visib',
              'pose', 'gender')]
    '''
    logger.info('Load raw AFLW dataset from "{}"'.format(sqlite_path))

    # Temporary dataset variables
    dataset_dict = dict()

    # Open sqlite file
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Basic property
    select_str = "faces.face_id, imgs.filepath, " \
                 "rect.x, rect.y, rect.w, rect.h, " \
                 "pose.roll, pose.pitch, pose.yaw, metadata.sex"
    from_str = "faces, faceimages imgs, facerect rect, facepose pose, " \
               "facemetadata metadata"
    where_str = "faces.file_id = imgs.file_id and " \
                "faces.face_id = rect.face_id and " \
                "faces.face_id = pose.face_id and " \
                "faces.face_id = metadata.face_id"
    query_res = _exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    for face_id, path, rectx, recty, rectw, recth, roll, pitch, yaw, gender\
            in query_res:
        # Data creation or conversion
        img_path = os.path.join(image_dir, path) if image_dir else path
        landmark = np.zeros((N_LANDMARK, 2), dtype=np.float32)
        landmark_visib = np.zeros(N_LANDMARK, dtype=np.float32)
        pose = np.array([roll, pitch, yaw], dtype=np.float32)
        gender = np.array(0 if gender == 'm' else 1, dtype=np.int32)
        others_landmark_pts = list()
        # Register
        data = {'face_id': face_id,
                'img_path': img_path,
                'rect': (rectx, recty, rectw, recth),
                'landmark': landmark,
                'landmark_visib': landmark_visib,
                'pose': pose,
                'gender': gender,
                'others_landmark_pts': others_landmark_pts}
        dataset_dict[face_id] = data

    # Landmark property
    # (Visibility is expressed by lack of the coordinate's row.)
    select_str = "faces.face_id, coords.feature_id, " \
                 "coords.x, coords.y"
    from_str = "faces, featurecoords coords"
    where_str = "faces.face_id = coords.face_id"
    query_res = _exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    invalid_face_ids = list()
    for face_id, feature_id, x, y in query_res:
        assert(1 <= feature_id <= N_LANDMARK)
        if face_id in dataset_dict:
            idx = feature_id - 1
            dataset_dict[face_id]['landmark'][idx][0] = x
            dataset_dict[face_id]['landmark'][idx][1] = y
            dataset_dict[face_id]['landmark_visib'][idx] = 1
        elif face_id not in invalid_face_ids:
            logger.warn('Invalid face id ({}) in AFLW'.format(face_id))
            invalid_face_ids.append(face_id)

    # Landmarks of other faces
    select_str = "a.face_id, coords.x, coords.y"
    from_str = "faces a, faces b, featurecoords coords"
    where_str = "a.face_id != b.face_id and a.file_id = b.file_id and " \
                "b.face_id = coords.face_id"
    query_res = _exec_sqlite_query(cursor, select_str, from_str, where_str)
    # Register to dataset_dict
    for face_id, others_x, others_y in query_res:
        if face_id in dataset_dict:
            other_coord = [others_x, others_y]
            dataset_dict[face_id]['others_landmark_pts'].append(other_coord)
        else:
            assert(face_id in invalid_face_ids)
    # Convert list to np.ndarray
    for data in dataset_dict.values():
        pts = np.array(data['others_landmark_pts'], dtype=np.float32)
        data['others_landmark_pts'] = pts

    # Exit sqlite
    cursor.close()

    # Return dataset_dict's value (list)
    return list(dataset_dict.values())


def _selective_search_dlib(img, max_img_size=(500, 500),
                           kvals=(50, 200, 2), min_size=2000, check=True,
                           debug_window=False):
    if debug_window:
        org_img = img

    # Resize the image for speed up
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
        resize_scale = org_h / h  # equal to `org_w / w`
        img = cv2.resize(img, (int(w), int(h)))

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


def _rect_contain(rect, pt):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def _extract_valid_rects(rects, landmark, visibility, img,
                         others_landmark_pts):
    ''' Extract rectangles which contain all landmark points or no them '''
    contained = False
    # Extraction
    dst = list()
    for rect in rects:
        mode = None  # t: all contained, f: no contained
        for pt, visib in zip(landmark, visibility):
            if visib == 0:
                continue
            if mode is None:
                # Set current mode
                mode = _rect_contain(rect, pt)
            else:
                # Check with mode
                if _rect_contain(rect, pt) != mode:
                    break
        else:
            # Check if others landmarks are contained
            for others_pt in others_landmark_pts:
                if _rect_contain(rect, others_pt):
                    break
            else:
                # No others landmark, and all contained or no contained
                if mode is True:
                    contained = True
                dst.append(rect)

    # Escape from no rectangle or no landmark
    if len(dst) == 0 or not contained:
        dst.append((0, 0, img.shape[1], img.shape[0]))

    return dst


def _rect_or(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def _rect_and(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return (0, 0, 0, 0)
    return (x, y, w, h)


def _rect_area(a):
    return a[2] * a[3]


def _rect_overlap_rate(a, b):
    area_and = _rect_area(_rect_and(a, b))
    area_or = _rect_area(_rect_or(a, b))
    if area_or == 0:
        return 0
    else:
        return math.sqrt(float(area_and) / float(area_or))


def _flip_y(img, landmark, landmark_visib, pose):
    # copy
    img = np.array(img)
    landmark = np.array(landmark)
    pose = np.array(pose)

    # Flip
    img = img[:, ::-1, :]

    # AFLW 21 points landmark
    #  0|LeftBrowLeftCorner
    #  1|LeftBrowCenter
    #  2|LeftBrowRightCorner
    #  3|RightBrowLeftCorner
    #  4|RightBrowCenter
    #  5|RightBrowRightCorner
    #  6|LeftEyeLeftCorner
    #  7|LeftEyeCenter
    #  8|LeftEyeRightCorner
    #  9|RightEyeLeftCorner
    #  10|RightEyeCenter
    #  11|RightEyeRightCorner
    #  12|LeftEar
    #  13|NoseLeft
    #  14|NoseCenter
    #  15|NoseRight
    #  16|RightEar
    #  17|MouthLeftCorner
    #  18|MouthCenter
    #  19|MouthRightCorner
    #  20|ChinCenter
    corres_dict = {
        0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0,  # brow
        6: 11, 7: 10, 8: 9, 9: 8, 10: 7, 11: 6,  # eye
        12: 16, 16: 12, 20: 20,  # ear and chin
        13: 15, 14: 14, 15: 13,  # nose
        17: 19, 18: 18, 19: 17  # mouse
    }
    dst_landmark = np.zeros_like(landmark)
    dst_landmark_visib = np.zeros_like(landmark_visib)
    for i, (pt, visib) in enumerate(zip(landmark, landmark_visib)):
        dst_i = corres_dict[i]
        dst_landmark[dst_i][0] = pt[0] * -1.0
        dst_landmark[dst_i][1] = pt[1]
        dst_landmark_visib[dst_i] = visib
    landmark = dst_landmark
    landmark_visib = dst_landmark_visib

    # pose (x, y, z), (roll, pitch yaw)
    pose[0] *= -1.0  # x
    pose[2] *= -1.0  # z

    return img, landmark, landmark_visib, pose


class ImageCacheLoader(object):

    def __init__(self, max_imgs_cnt=30000):
        self.cache = dict()
        self.max_imgs_cnt = max_imgs_cnt

    def __call__(self, path):
        try:
            return self.cache[path]
        except KeyError:
            img = cv2.imread(path)
            if len(self.cache) <= self.max_imgs_cnt:
                self.cache[path] = img
            else:
                logger.info('ImageCacheLoader is filled ({} images)'
                            .format(self.max_imgs_cnt))
            return img


class AFLW(chainer.dataset.DatasetMixin):
    ''' AFLW Dataset
    Arguments
        * n_try_detect_alt(int):
            The number of trying an alternation of the detection when
            get_example() is called. To disable the alternation, set 1.
        * overlap_tls(dict):
            Overlap thresholds of face rectangles and selective search's ones.
        * use_cache_loader(bool):
            A flag for use of cache loader to load images. Cache loader needs
            a lot of memory.
        ` random_flip(bool):
            A flag for random flip of datasets.
    '''

    def __init__(self, n_try_detect_alt=30,
                 overlap_tls={'detection_p': 0.50, 'detection_n': 0.35,
                              'landmark': 0.35, 'pose': 0.50, 'gender': 0.50},
                 use_cache_loader=False, random_flip=True):
        chainer.dataset.DatasetMixin.__init__(self)

        # Member variables
        self.dataset = list()
        self.detect_alt_diff = 0
        self.n_try_detect_alt = n_try_detect_alt
        self.overlap_tls = overlap_tls
        self.random_flip = random_flip

        # Image loader
        if use_cache_loader:
            self.load_img = ImageCacheLoader()
        else:
            self.load_img = cv2.imread

    def setup_raw(self, sqlite_path, image_dir, log_interval=10):
        # Load raw AFLW dataset
        self.dataset = _load_raw_aflw(sqlite_path, image_dir)

        # Calculate selective search rectangles (This takes many minutes)
        logger.info('Calculate selective search rectangles for AFLW')
        for i, entry in enumerate(self.dataset):
            # Logging
            if i % log_interval == 0:
                logger.info(' {}/{}'.format(i, len(self.dataset)))

            # Load image
            img = self.load_img(entry['img_path'])
            if img is None or img.size == 0:
                # Empty elements
                self.dataset[i]['ssrects'] = list()
                self.dataset[i]['ssrect_overlaps'] = list()
            else:
                # Selective search
                ssrects = _selective_search_dlib(img)
                ssrects = _extract_valid_rects(ssrects, entry['landmark'],
                                               entry['landmark_visib'], img,
                                               entry['others_landmark_pts'])
                overlaps = [_rect_overlap_rate(ssrect, entry['rect'])
                            for ssrect in ssrects]
                self.dataset[i]['ssrects'] = ssrects
                self.dataset[i]['ssrect_overlaps'] = overlaps

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        entry = self.dataset[i]

        # Check the number of selective search rectagles
        n_ssrects = len(entry['ssrects'])
        if n_ssrects == 0:
            logger.warn('No selective search rectangle')
            raise IndexError

        # Loop for detection alternation
        for _ in six.moves.xrange(self.n_try_detect_alt):
            # Random selective search rectangle to normalize
            ssrect_idx = random.randint(0, n_ssrects - 1)
            ssrect = entry['ssrects'][ssrect_idx]
            overlap = entry['ssrect_overlaps'][ssrect_idx]
            overlap = np.array(overlap, dtype=np.float32)

            # Detection
            if overlap > self.overlap_tls['detection_p']:
                detection = np.array(1, dtype=np.int32)
            elif overlap < self.overlap_tls['detection_n']:
                detection = np.array(0, dtype=np.int32)
            else:
                detection = np.array(-1, dtype=np.int32)  # Ignore

            # Check the alternation
            if detection == -1:
                continue
            # Negative sample
            if self.detect_alt_diff > 0 and detection == 0:
                self.detect_alt_diff -= 1
                break
            # Positive sample
            if self.detect_alt_diff <= 0 and detection == 1:
                self.detect_alt_diff += 1
                break

        # Entry variables
        img_path = entry['img_path']
        img = self.load_img(img_path)
        if img is None or img.size == 0:
            logger.warn('Invalid image "{}"'.format(img_path))
            raise IndexError
        landmark = entry['landmark']
        landmark_visib = entry['landmark_visib']
        pose = entry['pose']
        gender = entry['gender']

        # === Crop and Normalize ===
        x, y, w, h = ssrect
        # Image
        img = img[y:y + h + 1, x:x + w + 1, :]
        if img.size == 0:
            img = self.load_img(img_path)
            logger.warn('Invalid crop rectangle "{}" and image shape is "{}")'
                        .format(ssrect, img.shape))
            raise IndexError
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        # Landmark
        landmark_offset = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        landmark_denom = np.array([w, h], dtype=np.float32)
        landmark = (landmark - landmark_offset) / landmark_denom

        # === Random flip ===
        if self.random_flip and random.randint(0, 1):
            img, landmark, landmark_visib, pose = _flip_y(img, landmark,
                                                          landmark_visib, pose)

        # === Convert and Mask ===
        # Image
        img = np.transpose(img, (2, 0, 1))

        # Landmark and visibility
        if overlap > self.overlap_tls['landmark']:
            # [21, 2] -> [42]
            landmark = landmark.reshape(42)
            # Mask
            mask_landmark = np.ones_like(landmark)
            mask_landmark_visib = np.ones_like(landmark_visib)
        else:
            # [21, 2] -> [42]
            landmark = landmark.reshape(42)
            # No difference
            landmark = np.zeros_like(landmark)
            landmark_visib = np.zeros_like(landmark_visib)
            mask_landmark = np.zeros_like(landmark)
            mask_landmark_visib = np.zeros_like(landmark_visib)

        # Pose
        if overlap > self.overlap_tls['pose']:
            mask_pose = np.ones_like(pose)
        else:
            # No difference
            pose = np.zeros_like(pose)
            mask_pose = np.zeros_like(pose)

        # Gender
        if overlap > self.overlap_tls['gender']:
            pass
        else:
            gender = np.array(-1, dtype=np.int32)  # Ignore

        return {'x_img': img, 't_detection': detection, 't_landmark': landmark,
                't_visibility': landmark_visib, 't_pose': pose,
                't_gender': gender, 'm_landmark': mask_landmark,
                'm_visibility': mask_landmark_visib, 'm_pose': mask_pose}


def setup_aflw(cache_path, sqlite_path=None, image_dir=None, test_rate=0.04):
    # Empty AFLW
    aflw = AFLW()

    logger.info('Try to load AFLW cache from "{}"'.format(cache_path))
    try:
        # Load cache
        cache_data = np.load(cache_path)
        dataset = cache_data['dataset'].tolist()
        n_train = int(cache_data['n_train'])
        order = cache_data['order'].tolist()
        # Set to AFLW
        aflw.dataset = dataset
        logger.info('Succeeded in loading AFLW cache')

    except (FileNotFoundError, KeyError):
        # Setup AFLW
        logger.info('Failed to load AFLW cache, so setup now')
        if not sqlite_path:
            logger.critical('`sqlite_path` is needed to load raw AFLW')
        aflw.setup_raw(sqlite_path, image_dir)

        # Generate order to split into train/test
        n_train = int(len(aflw.dataset) * (1.0 - test_rate))
        order = np.random.permutation(len(aflw.dataset))

        # Save cache
        logger.info('Save AFLW cache to "{}"'.format(cache_path))
        np.savez(cache_path, dataset=aflw.dataset,
                 n_train=n_train, order=order)

    # Split dataset into train and test
    train, test = chainer.datasets.split_dataset(aflw, n_train, order=order)
    logger.info('AFLW datasets (n_train:{}, n_test:{})'.
                format(len(train), len(test)))

    return train, test
