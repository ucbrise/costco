import logging

BASIC_CONFIG = {
    "level": logging.INFO,
    "format": '[%(asctime)s] %(message)s',
    "datefmt": '%m/%d/%Y %H:%M:%S',
    "filemode": 'w',
}

STDERR_HANDLER = logging.StreamHandler()

def start_print_log():
    l = logging.getLogger()
    l.addHandler(STDERR_HANDLER)

def stop_print_log():
    l = logging.getLogger()
    l.removeHandler(STDERR_HANDLER)


