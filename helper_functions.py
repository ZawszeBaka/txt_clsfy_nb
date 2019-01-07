
import pandas as pd
import numpy as np

import os

from pprint import pprint

def get_data():

	cols = ['content','category']

	__is_init = True
	for csv_file in os.listdir('data'):
		if __is_init:
			data = pd.read_csv('data/'+csv_file)[cols]
			__is_init = False
		else:
			data = pd.concat([data, pd.read_csv('data/'+csv_file)[cols]])

	return data

def show_description(data):

	print(' [INFO] Shape ', data.shape)
	print(' [INFO] All categories ', data['category'].unique())

	# format [('Economy', 100)]
	num_articles_each_cat = []
	for cat in data['category'].unique():
		num_articles_each_cat.append( (cat ,data[data['category'] == cat].shape[0]))

	print(' [INFO] Amount articles of category : ')
	num_articles_each_cat = sorted(num_articles_each_cat, key = lambda x: x[1], reverse = True)
	pprint(num_articles_each_cat)

	print('\n\n')

def train_test_split(data, train_size = 0.8 ):


	train_loc = []
	test_loc = []

	for cat in data['category'].unique():
		indices = data[data['category'] == cat].index.values.tolist()
		num_train = int(len(indices)*0.8)
		train_loc += indices[:num_train]
		test_loc += indices[num_train:]

	train_set = data.iloc[train_loc,:]
	test_set = data.iloc[test_loc,:]

	train_set = train_set.reset_index(drop = True)
	test_set = test_set.reset_index(drop = True)

	return train_set, test_set

def choose_top_categories(data, TOP):
	num_articles_each_cat = []
	for cat in data['category'].unique():
		num_articles_each_cat.append( (cat ,data[data['category'] == cat].shape[0]))

	num_articles_each_cat = sorted(num_articles_each_cat, key = lambda x: x[1], reverse = True)
	pprint(num_articles_each_cat)

	chosen_cats = [p[0] for p in num_articles_each_cat[:TOP]]

	data = data.loc[data['category'].isin(chosen_cats)]
	data.reset_index(inplace = True, drop = True )

	print('\n [INFO] Filtered ! ')
	print(' [INFO] Shape ', data.shape)
	print(' [INFO] All categories : ', data['category'].unique())

	print(' [INFO] Data ', data.head())

	# format [('Economy', 100)]
	num_articles_each_cat = []
	for cat in data['category'].unique():
		num_articles_each_cat.append( (cat ,data[data['category'] == cat].shape[0]))

	print('\n [INFO] Amount articles of category : ')
	num_articles_each_cat = sorted(num_articles_each_cat, key = lambda x: x[1], reverse = True)
	pprint(num_articles_each_cat)

	return data


def calc_accuracy(y_test, predicted_y, info):
	size = len(y_test)
	right = 0
	for i in range(size):
		if y_test[i] == predicted_y[i]:
			right += 1

		# just for logging
		info['actual_category'].iloc[i] = y_test[i]

	info.to_csv('prediction.csv')
	return right/size
