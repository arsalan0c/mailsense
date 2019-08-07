#!/usr/bin/env python3

# This script evaluates a fastai model on a test set.

import argparse
from fastai import *
from fastai.text import *

parser = argparse.ArgumentParser(description='process arguments required for performing inference with trained model on input text')
parser.add_argument('-dp', '--datasetpath', help='str: path to csv dataset', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='str: path to output file containing model accuracy', type=str, action='store', required=True)
parser.add_argument('-md', '--modeldirectory', help='str: path to directory in which the saved model file resides', type=str, action='store', required=True)
parser.add_argument('-mn', '--modelname', help='str: file name of saved model', type=str, action='store', required=True)
args = parser.parse_args()

DATASET_COMPONENT_LABEL = 'set'
DATASET_TEXT_LABEL = 'text'
DATASET_TARGET_LABEL = 'polarity'

df = pd.read_csv(args.datasetpath)
test_df = df[df[DATASET_COMPONENT_LABEL] == 'test']

learn = load_learner(args.modeldirectory, args.modelname)
correct = 0
total = 0
for i in range(0, len(test_df)):
	to_predict = str(test_df.iloc[i][DATASET_TEXT_LABEL])
	target = str(test_df.iloc[i][DATASET_TARGET_LABEL])

	# get the polarity prediction
	result = str(learn.predict(to_predict)[0])
	if result == target:
		correct = correct + 1

	total = total + 1

accuracy = (correct / total * 100)
print('accuracy: ' + str(accuracy))
with open(args.outputpath, 'w') as f:
	f.write(str(accuracy))