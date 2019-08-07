#!/usr/bin/env python3

# This script modifies the emobank csv file to add sentiment orientation labels based on each text's valence value and removes unneeded columns

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-dr', '--datasetdir', help='string: directory location of data files', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='string: output path of the transformed csv data file (including name & extension)', type=str, action='store', required=True)
args = parser.parse_args()

df_emobank = pd.read_csv(args.datasetdir  + '/emobank.csv', index_col=[0])
# assign postive, negative and neutral labels based on valence values
df_emobank['polarity'] = df_emobank.apply(lambda row:
	'positive' if row.V > 3
	else 'negative' if row.V < 3
	else 'neutral',
	axis=1
)

df_processed_emobank = pd.DataFrame({'text': df_emobank.text, 'polarity': df_emobank.polarity})
df_amazon = pd.read_csv(args.datasetdir + '/amazon.csv', index_col=[0])
df_yelp = pd.read_csv(args.datasetdir + '/yelp.csv',  index_col=[0])
df_imdb = pd.read_csv(args.datasetdir + '/imdb.csv', index_col=[0])

data = [df_processed_emobank, df_amazon, df_yelp, df_imdb]
df = pd.concat(data)

# only keep alphabets and spaces
df['text'] = df['text'].str.replace('[^a-zA-Z ]', '')
df = df[(len(df.text) > 0) & (df.text != None) & ('NaN' not in df.text)]

# shuffle
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(args.outputpath)
print('Dataset Size:', len(df))
