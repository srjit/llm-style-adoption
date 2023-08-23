
import configparser
import os

__author__ = "Sreejith Sreekumar"
__email__ = "ssreejith@protonmail.com"
__version__ = "0.0.1"


config_file = "config.cfg"
config = None


def read():
    """
       Returns the external configurations for the program
    """

    global config
    if config is None:
        config = configparser.RawConfigParser()
        config.read(config.read(os.path.join(os.path.dirname(__file__), config_file)))
                                

    return config
