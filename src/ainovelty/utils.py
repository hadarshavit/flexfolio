import os
import logging


## http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
## Usage: Numbers = enum('ZERO', 'ONE', 'TWO') -> Numbers.ZERO = 0
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)



## http://stackoverflow.com/questions/2754126/python-logger-dynamic-filename
class MyFileHandler(object):

    def __init__(self, dir, logger, handlerFactory, **kw):
        kw['filename'] = os.path.join(dir, logger.name)
        self._handler = handlerFactory(**kw)

    def __getattr__(self, n):
        if hasattr(self._handler, n):
            return getattr(self._handler, n)
        raise AttributeError, n

