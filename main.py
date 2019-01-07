import pandas as pd
import numpy as np

import os

from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer

from helper_functions import *

from naive_bayes_discrete import  NB_discrete

## ================================================================================= ##


if __name__ == '__main__':

	# Get data from all csv file in 'data' directory
	data = get_data()

	## Choose 4 categories for classifying
	TOP = 4
	data = choose_top_categories(data, TOP)

	train_set, test_set = train_test_split(data, train_size = 0.8)

	print(' [INFO] Train set: ')
	show_description(train_set)

	print(' [INFO] Test set: ')
	show_description(test_set)

	print('data test', test_set.head())

	X_test = test_set['content'].tolist()
	y_test = test_set['category'].tolist()

	print(' [INFO] Train size ', len(train_set))
	print(' [INFO] Test size ', len(test_set))

	'''
		It fits data into the classifier
		It also filters out the word that contains a number
	'''
	clf = NB_discrete()
	clf.fit(train_set, p = 0.5, m = 3, manually_choose = False, scale = 1) # 800

	clf.debug()

	# predicted_y, info = clf.predict(X_test, is_show = False)
    #
	# print(' [INFO] Accuracy : ', calc_accuracy(y_test,predicted_y, info))

	## For improvement
