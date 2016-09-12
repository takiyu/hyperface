# -*- coding: utf-8 -*-
import logging


# default log format
default_fmt = logging.Formatter('[%(asctime)s] %(levelname)s '
                                '(%(process)d) %(name)s : %(message)s',
                                datefmt='%Y/%m/%d %H:%M:%S')

# set up handler
try:
    # Rainbow Logging
    import sys
    from rainbow_logging_handler import RainbowLoggingHandler
    default_handler = RainbowLoggingHandler(sys.stdout)
except:
    default_handler = logging.StreamHandler()

default_handler.setFormatter(default_fmt)
default_handler.setLevel(logging.DEBUG)

# setup root logger
logger = logging.getLogger()
logger.addHandler(default_handler)


def setFmt(fmt=default_fmt):
    global defaut_handler
    default_handler.setFormatter(fmt)


def setRootLevel(level):
    global logger
    logger.setLevel(level)
