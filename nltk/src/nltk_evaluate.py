#!/usr/bin/env python3

# This script evaluates a nltk sentiment intensity model on a test set.

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import pandas as pd
import argparse

import nltk_inference

parser = argparse.ArgumentParser(description='process arguments required for performing inference with trained model on input text')
parser.add_argument('-dp', '--datasetpath', help='str: path to csv dataset', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='str: path to output file containing model accuracy', type=str, action='store', required=True)
args = parser.parse_args()

DATASET_COMPONENT_LABEL = 'set'
DATASET_TEXT_LABEL = 'text'
DATASET_TARGET_LABEL = 'polarity'

df = pd.read_csv(args.datasetpath)
test_df = df[df[DATASET_COMPONENT_LABEL] == 'test']
test_df['text'] = test_df['text'].str.replace('[^a-zA-Z ]', '')

tokenized = test_df['text'].apply(lambda x: x.split())
tokenized = tokenized.apply(lambda x: [item for item in x if item not in stop_words])
tokenized = tokenized.reset_index(drop=True)

detokenized = []
for i in range(len(test_df)):
	t = ' '.join(tokenized[i])
	detokenized.append(t)

test_df['text'] = detokenized
print('Test Size: ' + str(len(test_df)))

nltk_inference.initialize_model()

correct = 0
total = 0
for i in range(0, len(test_df)):
	to_predict = str(test_df.iloc[i][DATASET_TEXT_LABEL])
	target = str(test_df.iloc[i][DATASET_TARGET_LABEL])

	# get the polarity prediction
	result = nltk_inference.predict(to_predict)
	if result == target:
		correct = correct + 1

	total = total + 1

accuracy = (correct / total * 100)
print('accuracy: ' + str(accuracy))

with open(args.outputpath, 'w') as f:
	f.write(str(accuracy))