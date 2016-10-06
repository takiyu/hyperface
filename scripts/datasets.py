# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
import random
import sqlite3

import chainer

import common

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


def _rect_contain(rect, pt):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def _extract_valid_rects(rects, img, others_landmark_pts):
    ''' Extract rectangles which do not contain other landmarks '''
    # Extraction
    dst = list()
    for rect in rects:
        # Check if others landmarks are contained
        for others_pt in others_landmark_pts:
            if _rect_contain(rect, others_pt):
                break
        else:
            dst.append(rect)

    # avoid no rectangle
    if len(dst) == 0:
        dst.append((0, 0, img.shape[1], img.shape[0]))

    return dst


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


class AFLW(chainer.dataset.DatasetMixin):
    ''' AFLW Dataset
    Arguments
        * n_try_detect_alt(int):
            The number of trying an alternation of the detection when
            get_example() is called. To disable the alternation, set 1.
        * overlap_tls(dict):
            Overlap thresholds of face rectangles and selective search's ones.
        * random_flip(bool):
            A flag for random flip of datasets.
        * min_valid_landmark_cnt(int):
            The least number for valid landmark. This value is used for
            deciding whether faces and their landmarks are valid.
    '''

    def __init__(self, n_try_detect_alt=30,
                 overlap_tls={'detection_p': 0.50, 'detection_n': 0.35,
                              'landmark': 0.35, 'pose': 0.50, 'gender': 0.50},
                 random_flip=True, min_valid_landmark_cnt=3):
        chainer.dataset.DatasetMixin.__init__(self)

        # Member variables
        self.dataset = list()
        self.detect_alt_diff = 0
        self.n_try_detect_alt = n_try_detect_alt
        self.overlap_tls = overlap_tls
        self.random_flip = random_flip
        self.min_valid_landmark_cnt = min_valid_landmark_cnt
        self.raw_mode = False

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
            img = cv2.imread(entry['img_path'])
            if img is None or img.size == 0:
                # Empty elements
                logger.warn('Failed to load image {}'.format(
                    entry['img_path']))
                self.dataset[i]['ssrects'] = list()
                self.dataset[i]['ssrect_overlaps'] = list()
            else:
                # Selective search
                ssrects = common.selective_search_dlib(img)
                ssrects = _extract_valid_rects(ssrects, img,
                                               entry['others_landmark_pts'])
                overlaps = [common.rect_overlap_rate(ssrect, entry['rect'])
                            for ssrect in ssrects]
                self.dataset[i]['ssrects'] = ssrects
                self.dataset[i]['ssrect_overlaps'] = overlaps

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        entry = self.dataset[i]

        # Raw entry
        if self.raw_mode:
            return entry

        # Loop for detection alternation
        try_cnt = 0
        special_skip_cnt = 0
        while try_cnt < self.n_try_detect_alt:
            try_cnt += 1

            # === Entry variables ===
            img_path = entry['img_path']
            landmark = entry['landmark']
            landmark_visib = entry['landmark_visib']
            pose = entry['pose']
            gender = entry['gender']
            ssrects = entry['ssrects']
            overlaps = entry['ssrect_overlaps']
            # Random ssrect
            if len(ssrects) == 0:
                logger.warn('No selective search rectangle')
                raise IndexError
            ssrect_idx = random.randint(0, len(ssrects) - 1)
            ssrect = ssrects[ssrect_idx]
            overlap = overlaps[ssrect_idx]
            x, y, w, h = ssrect

            # === Crop and Normalize 1 (landmark) ===
            # Landmark ([-0.5:0.5])
            landmark_offset = np.array(
                [x + w / 2, y + h / 2], dtype=np.float32)
            landmark_denom = np.array([w, h], dtype=np.float32)
            landmark = (landmark - landmark_offset) / landmark_denom
            # Consider range of the cropped rectangle
            hidden_idxs = np.where((landmark < -0.5) | (0.5 < landmark))
            landmark = landmark.copy()
            landmark[hidden_idxs[0]] = 0.0  # mask [x, y] (overwrite)
            landmark_visib = landmark_visib.copy()
            landmark_visib[hidden_idxs[0]] = 0.0  # mask (overwrite)

            # === Convert 1 (Detection) ===
            if overlap > self.overlap_tls['detection_p']:
                detection = np.array(1, dtype=np.int32)
            elif overlap < self.overlap_tls['detection_n']:
                detection = np.array(0, dtype=np.int32)
            else:
                detection = np.array(-1, dtype=np.int32)  # Ignore

            # === Special Skip for invalid landmark faces ===
            n_valid_landmark = landmark_visib[landmark_visib > 0.5].shape[0]
            if detection == 1 and \
               n_valid_landmark < self.min_valid_landmark_cnt:
                try_cnt -= 1
                special_skip_cnt += 1
                if special_skip_cnt > 10:  # avoid infinity loop
                    break
                else:
                    continue

            # === Check the alternation ===
            if detection == -1:
                continue
            if self.detect_alt_diff > 0 and detection == 0:  # Negative sample
                self.detect_alt_diff -= 1
                break
            if self.detect_alt_diff <= 0 and detection == 1:  # Positive sample
                self.detect_alt_diff += 1
                break
        # End of detection alternation loop

        # === Crop and Normalize 2 (image) ===
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            logger.warn('Invalid image "{}"'.format(img_path))
            raise IndexError
        # Image
        img = img[y:y + h + 1, x:x + w + 1, :]
        if img.size == 0:
            org_img = cv2.imread(img_path)
            logger.warn('Invalid crop rectangle. (rect:{}, img:{}, org_img{}")'
                        .format(ssrect, img.shape, org_img.shape))
            raise IndexError
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32)
        # [0:255](about) -> [-0.5:0.5]
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)

        # === Random flip ===
        if self.random_flip and random.randint(0, 1):
            flip_ret = _flip_y(img, landmark, landmark_visib, pose)
            img, landmark, landmark_visib, pose = flip_ret

        # === Convert 2 (and mask) ===
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
            pass  # use entry value
        else:
            gender = np.array(-1, dtype=np.int32)  # Ignore

        return {'x_img': img, 't_detection': detection, 't_landmark': landmark,
                't_visibility': landmark_visib, 't_pose': pose,
                't_gender': gender, 'm_landmark': mask_landmark,
                'm_visibility': mask_landmark_visib, 'm_pose': mask_pose}


def setup_aflw(cache_path, sqlite_path=None, image_dir=None, test_rate=0.04,
               raw_mode=False):
    # Empty AFLW
    aflw = AFLW()
    aflw.raw_mode = raw_mode

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


def aflw_template_landmark():
    ''' Template landmark from AFLW dataset '''
    return np.array([[-0.479962468147, 0.471864163876],
                     [-0.30303606391, 0.508996844292],
                     [-0.106451146305, 0.498075485229],
                     [0.106451146305, 0.498075485229],
                     [0.30303606391, 0.508996844292],
                     [0.479962468147, 0.471864163876],
                     [-0.447198301554, 0.321149080992],
                     [-0.318325966597, 0.325517624617],
                     [-0.163242310286, 0.308043420315],
                     [0.163242310286, 0.308043420315],
                     [0.318325966597, 0.325517624617],
                     [0.447198301554, 0.321149080992],
                     [-0.674257874489, -0.151652157307],
                     [-0.170000001788, -0.075740583241],
                     [0.0, 0.0],
                     [0.170000001788, -0.075740583241],
                     [0.674257874489, -0.151652157307],
                     [-0.272456139326, -0.347239643335],
                     [0.0, -0.336318254471],
                     [0.272456139326, -0.347239643335],
                     [0.0, -0.737950384617]], dtype=np.float32)
