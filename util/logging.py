import logging
import logging.handlers
import sys


def set_logger(name, level, log_furl=None):
    """
    Creates/retrieves a Logger object with the desired name and level.
    """
    logger = logging.getLogger(name)

    if not len(logger.handlers):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if log_furl is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.handlers.RotatingFileHandler(log_furl, maxBytes=10 * 1024 * 1024, backupCount=5)

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    logger.setLevel(level_dict[level])

    return logger


def get_logger(name):
    return logging.getLogger(name)


