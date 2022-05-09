import os
import datetime

COMMON_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
PROJECT_PATH = os.path.abspath(COMMON_PATH + '..') + '/'
HEIGHT_MAPS_PATH = os.path.abspath(PROJECT_PATH + 'height_maps') + '/'
SCRIPTS_PATH = os.path.abspath(PROJECT_PATH + 'scripts/') + '/'

now = datetime.datetime.now()
CURRENT_DATETIME_STR = str(now)[:10] + '-' + str(now)[11:13] + '-' + str(now)[14:16] + '-' + str(now)[17:19]
