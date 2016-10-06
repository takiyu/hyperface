# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as agg
except Exception as e:
    logger.error('Failed to import matplotlib')
    logger.error('[%s] %s', str(type(e)), str(e.args))
    exit()


def _draw_line(img, pt1, pt2, color, thickness=2):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, int(thickness))


def _draw_circle(img, pt, color, radius=4, thickness=-1):
    pt = (int(pt[0]), int(pt[1]))
    cv2.circle(img, pt, radius, color, int(thickness))


def _draw_rect(img, rect, color, thickness=2):
    p1 = (int(rect[0]), int(rect[1]))
    p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
    cv2.rectangle(img, p1, p2, color, thickness)


def _draw_cross(img, pt, color, size=4, thickness=2):
    p0 = (pt[0] - size, pt[1] - size)
    p1 = (pt[0] + size, pt[1] + size)
    p2 = (pt[0] + size, pt[1] - size)
    p3 = (pt[0] - size, pt[1] + size)
    _draw_line(img, p0, p1, color, thickness)
    _draw_line(img, p2, p3, color, thickness)


def _rotation_matrix(rad_x, rad_y, rad_z):
    cosx, cosy, cosz = math.cos(rad_x), math.cos(rad_y), math.cos(rad_z)
    sinx, siny, sinz = math.sin(rad_x), math.sin(rad_y), math.sin(rad_z)
    rotz = np.array([[cosz, -sinz, 0],
                     [sinz, cosz, 0],
                     [0, 0, 1]], dtype=np.float32)
    roty = np.array([[cosy, 0, siny],
                     [0, 1, 0],
                     [-siny, 0, cosy]], dtype=np.float32)
    rotx = np.array([[1, 0, 0],
                     [0, cosx, -sinx],
                     [0, sinx, cosx]], dtype=np.float32)
    return rotx.dot(roty).dot(rotz)


def _project_plane_yz(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    return np.array([x, -y], dtype=np.float32)  # y flip


def draw_detection(img, detection, size=15):
    # Upper left
    pt = (size + 5, size + 5)
    if detection:
        _draw_circle(img, pt, (0, 0.7, 0), size, 5)
    else:
        _draw_cross(img, pt, (0, 0, 0.7), size, 5)


def draw_landmark(img, landmark, visibility, color, line_color_scale,
                  denormalize_scale=True):
    """  Draw AFLW 21 points landmark
        0|LeftBrowLeftCorner
        1|LeftBrowCenter
        2|LeftBrowRightCorner
        3|RightBrowLeftCorner
        4|RightBrowCenter
        5|RightBrowRightCorner
        6|LeftEyeLeftCorner
        7|LeftEyeCenter
        8|LeftEyeRightCorner
        9|RightEyeLeftCorner
        10|RightEyeCenter
        11|RightEyeRightCorner
        12|LeftEar
        13|NoseLeft
        14|NoseCenter
        15|NoseRight
        16|RightEar
        17|MouthLeftCorner
        18|MouthCenter
        19|MouthRightCorner
        20|ChinCenter
    """
    conn_list = [[0, 1], [1, 2], [3, 4], [4, 5],  # brow
                 [6, 7], [7, 8], [9, 10], [10, 11],  # eye
                 [13, 14], [14, 15], [13, 15],  # nose
                 [17, 18], [18, 19],  # mouse
                 [12, 20], [16, 20]]  # face contour

    if landmark.ndim == 1:
        landmark = landmark.reshape(int(landmark.shape[-1] / 2), 2)
    assert(landmark.shape[0] == 21 and visibility.shape[0] == 21)

    if denormalize_scale:
        h, w = img.shape[0:2]
        size = np.array([[w, h]], dtype=np.float32)
        landmark = landmark * size + size / 2

    # Line
    line_color = tuple(v * line_color_scale for v in color)
    for i0, i1 in conn_list:
        if visibility[i0] > 0.5 and visibility[i1] > 0.5:
            _draw_line(img, landmark[i0], landmark[i1], line_color, 2)

    # Point
    for pt, visib in zip(landmark, visibility):
        if visib > 0.5:
            _draw_circle(img, pt, color, 4, -1)
        else:
            _draw_circle(img, pt, color, 4, 1)


def draw_pose(img, pose, size=30, idx=0):
    # parallel projection (something wrong?)
    rotmat = _rotation_matrix(-pose[0], -pose[1], -pose[2])
    zvec = np.array([0, 0, 1], np.float32)
    yvec = np.array([0, 1, 0], np.float32)
    xvec = np.array([1, 0, 0], np.float32)
    zvec = _project_plane_yz(rotmat.dot(zvec))
    yvec = _project_plane_yz(rotmat.dot(yvec))
    xvec = _project_plane_yz(rotmat.dot(xvec))

    # Lower left
    org_pt = ((size + 5) * (2 * idx + 1), img.shape[0] - size - 5)
    _draw_line(img, org_pt, org_pt + zvec * size, (1, 0, 0), 3)
    _draw_line(img, org_pt, org_pt + yvec * size, (0, 1, 0), 3)
    _draw_line(img, org_pt, org_pt + xvec * size, (0, 0, 1), 3)


def draw_gender(img, gender, size=7, idx=0):
    # Upper right
    pt = (img.shape[1] - (size + 5) * (2 * idx + 1), size + 5)
    if gender == 0:
        _draw_circle(img, pt, (1.0, 0.3, 0.3), size, -1)  # male
    elif gender == 1:
        _draw_circle(img, pt, (0.3, 0.3, 1.0), size, -1)  # female


def draw_gender_rect(img, gender, rect):
    if gender == 0:
        _draw_rect(img, rect, (1.0, 0.3, 0.3))  # male
    elif gender == 1:
        _draw_rect(img, rect, (0.3, 0.3, 1.0))  # female


def draw_loss_graph(train_loss_list, test_loss_list, train_epoch_list=None,
                    test_epoch_list=None, train_color='blue', test_color='red',
                    legend_loc='upper right', title=None):
    # Axis data
    # Losses
    train_loss = np.asarray(train_loss_list)
    test_loss = np.asarray(test_loss_list)
    # Epochs
    if train_epoch_list:
        train_epoch = np.asarray(train_epoch_list)
    else:
        train_epoch = np.arange(0, len(train_loss_list))
    if test_epoch_list:
        test_epoch = np.asarray(test_epoch_list)
    else:
        test_epoch = np.arange(0, len(test_loss_list))

    # Create new figure
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(train_epoch, train_loss, label='train', color=train_color)
    ax.plot(test_epoch, test_loss, label='test', color=test_color)

    def draw_annotate(label, x, y, color):
        ax.scatter(x, y, 20, color=color)
        ax.annotate(label, xy=(x, y), xytext=(+20, +10),
                    textcoords='offset points',
                    arrowprops={'arrowstyle': '->',
                                'connectionstyle': 'arc3,rad=.2'})

    # Show min values
    if train_loss.shape[0] > 0:
        min_idx = np.argmin(train_loss)
        x, y = train_epoch[min_idx], train_loss[min_idx]
        draw_annotate('min train loss: %0.3f' % y, x, y, train_color)
    if test_loss.shape[0] > 0:
        min_idx = np.argmin(test_loss)
        x, y = test_epoch[min_idx], test_loss[min_idx]
        draw_annotate('min test loss: %0.3f' % y, x, y, test_color)

    # Settings
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss rate")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title)

    # Draw
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    img = np.fromstring(renderer.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    # Close
    plt.close('all')

    return img
