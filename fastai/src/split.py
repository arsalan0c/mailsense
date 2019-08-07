#!/usr/bin/env python3

# This script splits a csv dataset into train, val and test sets.

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--datasetpath', help='string: path to the csv data file', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='string: output path of the split csv data file (including name & extension)', type=str, action='store', required=True)
parser.add_argument('-train', '--trainsize', help='float: size of training set as a proportion of total.', type=float, action='store', default=0.7)
parser.add_argument('-test', '--testsize', help='float: size of test set as a proportion of total.', type=float, action='store', default=0.1)
args = parser.parse_args()

DATASET_COMPONENT_LABEL = 'set'

df = pd.read_csv(args.datasetpath, index_col=[0])
train_val_df, test_df = train_test_split(df, test_size=args.testsize)
valsize = 1.0 - args.trainsize - args.testsize
assert valsize >= 0.0
train_df, val_df = train_test_split(train_val_df, test_size=valsize/(args.trainsize + valsize))
assert len(train_df) + len(val_df) + len(test_df) == len(df)

# set labels
train_df[DATASET_COMPONENT_LABEL] = 'train'
val_df[DATASET_COMPONENT_LABEL] = 'val'
test_df[DATASET_COMPONENT_LABEL] = 'test'

print('Train Size:', len(train_df))
print('Validation Size:', len(val_df))
print('Test Size:', len(test_df))

df_s = [train_df, val_df, test_df]
df_merged = pd.concat(df_s, ignore_index=True)
columns = df_merged.columns
# df_merged.drop([columns[1]], axis=1, inplace=True)
df_merged = df_merged.dropna()
df_merged.to_csv(args.outputpath)