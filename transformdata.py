#!/usr/bin/env python3

# This script rmodifies the emobank csv file to add sentiment orientation labels based on each text's valence value and removes unneeded columns

import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-dp', '--datasetpath', help='string: path to the csv data file', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='string: output path of the transformed csv data file (including name & extension)', type=str, action='store', required=True)
args = parser.parse_args()

df = pd.read_csv(args.datasetpath)
# assign postive, negative and neutral labels based on valence values
df['polarity'] = df.apply(lambda row:
	'positive' if row.V > 3
	else 'negative' if row.V != 3
	else 'neutral',
	axis=1
)

# columns to remove from dataframe
to_drop = ['id', 'V', 'A', 'D']
df.drop(to_drop, inplace=True, axis=1)
# shuffle
df = df.sample(frac=1).reset_index(drop=True)
# save
df.to_csv(args.outputpath)

