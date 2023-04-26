import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - $(levelname)s - $(message)s', level=logging.INFO)
logging.info('Start')

DATA_DIR = './data/CamVid/'
# Load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    logging.info('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    logging.info('Done!')

logging.info('Done!')