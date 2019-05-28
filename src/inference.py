#!/usr/bin/env python3

import argparse
from fastai import *
from fastai.text import *

parser = argparse.ArgumentParser(description='process arguments required for performing inference with trained model on input text')
parser.add_argument('-t', '--text', help='string: text to perform inference on', type=str, action='store', required=True)
args = parser.parse_args()

MODEL_DIR = 'models'
MODEL_NAME = 'textclassifier_encoder'

learn = load_learner(MODEL_DIR, MODEL_NAME)
result = learn.predict(args.text)