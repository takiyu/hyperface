# -*- coding: utf-8 -*-
import base64
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import multiprocessing
import os
import threading
import types


# logging
from logging import getLogger, NullHandler, CRITICAL
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# disable werkzeug logger
werkzeug_logger = getLogger('werkzeug')
werkzeug_logger.setLevel(CRITICAL)
# disable engineio logger
engineio_logger = getLogger('engineio')
engineio_logger.setLevel(CRITICAL)
# disable socketio logger
socketio_logger = getLogger('socketio')
socketio_logger.setLevel(CRITICAL)


IO_NAMESPACE = '/viewer'


def encodeimg(img, ext='.jpeg'):
    try:
        ret, img = cv2.imencode(ext, img)
        if not ret:
            raise
        img = img.tostring()
        img = base64.encodestring(img)
        img = 'data:image/jpeg;base64,' + img.decode('ascii')
        return img
    except Exception:
        logger.error('Failed to encodeimg()')
        return None


def setImgWidthElement(data, img_key, width_key, resize=True):
    ''' Set width in dict data
    If the width is smaller than actual width and `resize` is True,
    image will be resized.
    '''
    try:
        img = data[img_key]
    except KeyError:
        logger.error('No image data (key: %s)' % img_key)
        return

    try:
        width = data[width_key]
        # resize to make the image smaller
        if resize and width < img.shape[1]:
            height = int(width * img.shape[0] / img.shape[1])
            logger.debug('Resize to (%d, %d)' % (width, height))
            data[img_key] = cv2.resize(img, (width, height))
    except KeyError:
        width = img.shape[1]
        # set width
        data[width_key] = width


def encodeImgElement(data, key):
    ''' Encode image in dict data '''
    try:
        img = encodeimg(data[key])
        if img is None:
            raise Exception()
        data[key] = img
    except KeyError:
        logger.error('No image data (key: %s)' % key)
    except:
        logger.error('Invalid image data (key: %s)' % key)
        try:
            data.pop(key)
        except:
            pass


class ImageBufferingThread(threading.Thread):

    def __init__(self, input_queue):
        threading.Thread.__init__(self)
        self.pool = {}
        self.input_queue = input_queue
        self.update_event = None

    def run(self):
        while True:
            # wait for input
            tab, name, data = self.input_queue.get()

            # set to pool
            if name is None:
                # delete tab
                self.delete_tab(tab)
            elif data is None:
                # delete buffer data
                self.delete_one(tab, name)
            else:
                # set image size
                setImgWidthElement(data, 'img', 'width')
                # encode image
                encodeImgElement(data, 'img')
                # update buffer data
                self.set_one(tab, name, data)
            # call update event
            if self.update_event is not None:
                logger.debug('Call update_event()')
                self.update_event(tab, name, data)

    def delete_tab(self, tab):
        try:
            logger.debug('Delete buffer data (tab: %s)' % str(tab))
            self.pool.pop(tab)
        except:
            logger.error('Failed to delete buffer (tab: %s)' % str(tab))

    def delete_one(self, tab, name):
        try:
            logger.debug('Delete buffer data (tab: %s, name: %s)' %
                         (str(tab), str(name)))
            self.pool[tab].pop(name)
        except:
            logger.error('Failed to delete buffer (tab: %s, name: %s)' %
                         (str(tab), str(name)))

    def set_one(self, tab, name, data):
        try:
            logger.debug('Update buffer data (tab: %s, name: %s)' %
                         (str(tab), str(name)))
            # create tab
            try:
                self.pool[tab]
            except KeyError:
                self.pool[tab] = {}
            # set data
            self.pool[tab][name] = data
        except:
            logger.error('Failed to update buffer (tab: %s, name: %s)' %
                         (str(tab), str(name)))

    def get_data_all(self):
        dst = list()
        for tab, imgs in self.pool.items():
            for name, data in imgs.items():
                dst.append((tab, name, data))
        return dst

    def register_update_event_func(self, update_event):
        if isinstance(update_event, types.FunctionType) or \
           update_event is None:
            self.update_event = update_event
        else:
            logger.error('Update event must be a function or None')


def new_server(viewer_queue, stop_page, port, secret_key):
    # create server
    app = Flask(__name__, static_url_path='/static')
    app.config['SECRET_KEY'] = secret_key
    # must be 'threading' for broadcast emitting
    socketio = SocketIO(app, async_mode='threading',
                        logger=False, engineio_logger=False)

    # rooting
    @app.route('/')
    def __index():
        logger.info('Render viewer page')
        return render_template('index.html', script="index.js")

    if stop_page:
        @app.route('/stop')
        def __stop():
            socketio.stop()
            logger.info('Server stop request')
            return 'This server is stopped'

    @socketio.on('connect', namespace=IO_NAMESPACE)
    def __on_viewer_connect():
        logger.info('New viewer connection is established')

    @socketio.on('disconnect', namespace=IO_NAMESPACE)
    def __on_viewer_disconnect():
        logger.info('Viewer connection is closed')

    @socketio.on('update', namespace=IO_NAMESPACE)
    def __on_update():
        logger.info('Image updating request is received')
        # get all of current data
        emit_data = buffering_thread.get_data_all()
        # emit all
        logger.debug('Emit for update all')
        emit('update', emit_data, namespace=IO_NAMESPACE)

    def update_event(tab, name, data):
        emit_data = [[tab, name, data]]  # single data
        # broadcast emit
        logger.debug('Broadcast emit for update (tab: %s, name: %s)' %
                     (str(tab), str(name)))
        socketio.emit('update', emit_data, namespace=IO_NAMESPACE)

    # create image updating thread
    if viewer_queue:
        logger.info('Start image buffering thread')
        buffering_thread = ImageBufferingThread(viewer_queue)
        buffering_thread.daemon = True
        buffering_thread.start()
        buffering_thread.register_update_event_func(update_event)

    # start server
    logger.info('Start server on port %d' % port)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    logger.info('Stop server on port %d' % port)


def start(viewer_queue, stop_page=True, port=5000, secret_key=os.urandom(24)):
    process = multiprocessing.Process(target=new_server,
                                      args=(viewer_queue, stop_page,
                                            port, secret_key))
    process.daemon = True
    process.start()
