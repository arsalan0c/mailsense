#!/bin/bash

# use relative path
cd $(dirname $0)/../source/

# get sentiment labelled sentences
wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip'
unzip '../source/sentiment labelled sentences.zip'
cd 'sentiment labelled sentences'
for filename in *; do 
	# check if it is a data file
	if [[ $filename == *'_'* ]]; then
		output_name=${filename%%_*}
		output_path="../$output_name.csv"
		python3 ../../src/process_sls.py -dp $filename -op $output_path
	fi
done

cd ../
# get emobank
cp EmoBank/corpus/emobank.csv ./
