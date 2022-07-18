import os
import sys
import shutil
import logging
from logging import INFO, ERROR, DEBUG, CRITICAL, FATAL, WARNING, WARN, NOTSET

import datetime
from pathlib import Path


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': RED,
    'ERROR': RED
}

class Logger(logging.Logger):
    def __init__(self, path_output, path_config=None, postfix=None, level=logging.NOTSET ):
        logging.Logger.__init__(self, '', level)

        # make dir
        basename = path_config.name.split(".")[0] if path_config is not None else "log"
        if postfix is None or "":
            postfix = f"{datetime.datetime.now().strftime('%Y%m%d-%HH%MM%SS')}" 

        self.path_output = Path(path_output).joinpath(basename, postfix)
        self.path_output.mkdir(parents=True, exist_ok=True)

        # copy config
        dst = self.path_output.joinpath(path_config.name)
        shutil.copyfile(src=path_config, dst=dst)

        # add file hander
        filename = self.path_output.joinpath(f"logger.log")
        fHander = logging.FileHandler(filename)
        fFormat = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        fHander.setFormatter(fFormat)
        self.addHandler(fHander)

        # add console hander
        cHander = logging.StreamHandler(sys.stdout)
        cFormat = ColoredFormatter( RESET_SEQ+"[%(asctime)s] %(message)s"+RESET_SEQ )
        cHander.setFormatter(cFormat)
        self.addHandler(cHander)
    def setContext(self, key, value):
        self.context[key] = value

    def log(self, msg,  level, *args, **kwargs):
        extra = kwargs
        extra.update(self.context)
        # super(Logger, self).info(msg)

    def error(self, msg, *args, **kwargs):
        super(Logger, self).error(msg, *args, **kwargs)

    def exception(self, *args, **kwargs):
        e = sys.exc_info()
        extra = kwargs
        extra.update(self.context)
        super(Logger, self).error(e[1].message)



class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg, datefmt='%Y-%m-%d %H:%M:%S')
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30+COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)
