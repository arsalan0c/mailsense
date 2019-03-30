from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def load_data(path):
	df = pd.read_csv(path)
	# split into train + val AND test
	train_val_df, test_df = train_test_split(df, test_size=0.1)
	# split train + val into train and val. train: 72%, val: 18%, test: 10%
	train_df, val_df = train_test_split(train_val_df, test_size=0.20)

	# Language model data
	data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, text_cols=1, label_cols=2, path="")
	# Classifier model data
	data_clas = TextClasDataBunch.from_df(train_df=train_df, valid_df=val_df, test_df=test_df, vocab=data_lm.train_ds.vocab, bs=32, text_cols=1, label_cols=2, path="")
	return data_lm, data_clas, test_df

def train(data_lm, data_clas, lm_epochs=2, tc_epochs=2):
	# create language model with pretrained weights
	learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.5)
	# train language model using one cycle policy
	learn.fit_one_cycle(lm_epochs, 1e-2)
	learn.save_encoder('lm_enc')
	learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
	learn.load_encoder('lm_enc')
	learn.fit_one_cycle(tc_epochs, 1e-2)
	return learn

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-dp', '--datasetpath', help='str: path to csv dataset', type=str, action='store', required=True)
	parser.add_argument('-lme', '--learningmodelepochs', help='int: number of epochs to train the language model learner', type=int, action='store', default=2)
	parser.add_argument('-tce', '--textclassifierepochs', help='int: number of epochs to train the text classifier', type=int, action='store', default=2)
	args = parser.parse_args()

	data_lm, data_clas, test_df = load_data(args.datasetpath)
	learn = train(data_lm, data_clas, args.learningmodelepochs, args.textclassifierepochs)
	learn.save_encoder('tc_enc')
	preds, y, losses = learn.get_preds(DatasetType.Test, with_loss=True)
	print(preds)
	interp = ClassificationInterpretation(learn, preds, y, losses)
	print(interp.confusion_matrix(slice_size=10))
	print(learn.predict('hello there, my name is Arsalan. How are you doing today?'))
	pred_df = test_df
	pred_df['polarity'] = pred_df['text'].apply(lambda row: str(learn.predict(row)[0]))
	pred_df['text'] = pred_df['text'].apply(lambda row: row)
	pred_df.to_csv('results.csv')

if __name__ == '__main__':
	main()









