#!/usr/bin/env python3

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process arguments to convert text files of the sentiment labelled sentences dataset to csv files')
parser.add_argument('-dp', '--datasetpath', help='string: path of data .txt file', type=str, action='store', required=True)
parser.add_argument('-op', '--outputpath', help='string: output path of .csv file', type=str, action='store', required=True)
args = parser.parse_args()

with open(args.datasetpath) as f:
	lines = f.readlines()
	lines = [x.strip() for x in lines]
	
	processed_lines = []
	polarities = []
	for line in lines:
		polarity = None
		processed_line = line
		if len(line) > 0:
			polarity = "positive" if line[len(line) - 1] == "1" else "negative"
			processed_line = line[:-1]

		if polarity is not None:
			processed_lines.append(processed_line)
			polarities.append(polarity)

	df = pd.DataFrame(data={'text': processed_lines, 'polarity': polarities})
	df.to_csv(args.outputpath)
