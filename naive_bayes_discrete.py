
import pandas as pd
import numpy as np
import math

from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer

## following the HW requirement
class NB_discrete():

	def fit(self,data, p = 0.5, m = 3, manually_choose = False, scale = 100):
		'''
			data is a type of DataFrame
			data contains 2 cols ['content', 'category']
		'''

		vectorizer = CountVectorizer()
		print('\n [INFO] Vectorizer ')
		print(vectorizer)

		# Vectorizing ...
		print('\n [INFO] Vectorizing ... ', end ='')
		corpus = data['content']
		X = vectorizer.fit_transform(corpus).toarray()
		print(' Done ! ')
		print(' [INFO] ', X.shape[0], ' vectors in ', X.shape[1], '-th-dimensional')

		feature_names = vectorizer.get_feature_names()

		# Filter out the string which contains a number
		X, good_feature_names = self.filter_out_number(X,feature_names)

		log = pd.DataFrame(X, columns = good_feature_names)
		print(' [INFO] After filtering out the string which contains a number ', len(good_feature_names))

		# for save
		log = log.iloc[:100,500:700]
		log['#content'] = corpus[:100]
		print(' [INFO] Saving data after vectorizing to data.csv ...' , end='')
		log.to_csv('data_brief.csv')
		print(' Done !')
		pd.DataFrame(good_feature_names, columns = ['feature_names']).to_csv('feature_names.csv')

		# Store values !!
		# self.data contains columns ['word_1', 'word_2',... , '#CAT']
		self.data = pd.DataFrame(X, columns = good_feature_names)
		self.data['#CAT'] = data['category']

		if not manually_choose:
			m = np.float64(data.columns.values.shape[0]-1)
			p = np.float64(1)
		else:
			m = np.float64(m)
			p = np.float64(p)

		def g_f(data):
			return data[data.columns.values[:-1]]

		def calc(n_c, n):
			return (n_c + m*p)/(n+m) * scale

		# processing data
		"""
			n_c + m*p
			---------
			  n + m

		count_dict format :
			{
				'label_1': {
					'#COUNT':  ,
					'#PROB' :  ,
					'word_1':  ,
					'word_2':  ,
					...
					'#NOTEXIST':
				},
				...
			}
		label_words format:
			{
				'label_1: ['word_1', 'word_2', ... ],
				...
			}
		"""
		self.count_dict = dict()
		self.label_words = dict()

		data = self.data
		words = g_f(data).columns.values
		labels = data['#CAT'].unique()

		print(' [INFO] Fitting data ... ', end='')
		for label in labels:

			filtered_data = data[data['#CAT'] == label]

			label_word = []

			# count this label
			each_lb = dict()
			each_lb['#COUNT'] = np.float64(g_f(filtered_data).sum().sum())

			# probability of this label
			each_lb['#PROB'] = np.float64(filtered_data.shape[0] / data.shape[0])

			#iterate through each word
			for w in words:
				n_c = np.float64(filtered_data[w].sum())
				if n_c != 0.0 :
					each_lb[w] = calc(n_c, each_lb['#COUNT'])
					label_word.append(w)

			each_lb['#NOTEXIST'] = calc(0.0, each_lb['#COUNT'])

			self.count_dict[label] = each_lb
			self.label_words[label] = label_word

		print(' Done !')


	def filter_out_number(self, X, feature_names):
		# Check if a string contains a number
		def has_number(st):
			return any(char.isdigit() for char in st)

		# Filter out the string which contains a number
		pos = []
		i = 0
		good_feature_names = []
		for f in feature_names:
			if not has_number(f):
				good_feature_names.append(f)
				pos.append(i)
			i += 1

		X = X[:,pos]

		return X, good_feature_names

	def get_prob_per_label(data):
		rs = dict()
		for cat in data['#CAT'].unique():
			rs[cat] = data[data['#CAT'] == cat].shape[0]
		return rs

	def predict(self,test, is_show = False):
		'''
			Array or list
		'''
		if not is_show:
			print(' [INFO] Predicting ... ', end = '')

		data = self.data

		# Check if a string contains a number
		def has_number(st):
			return any(char.isdigit() for char in st)

		labels = data['#CAT'].unique()
		features = data.columns.values

		predicted_cls = []

		info = pd.DataFrame(columns=labels.tolist() + ['content', 'predicted_category', 'actual_category' ])

		for raw in test:
			# Vectorizing
			vectorizer = CountVectorizer()
			X = vectorizer.fit_transform([raw]).toarray()
			feature_names = vectorizer.get_feature_names()

			# Filter out the string that contains a number
			X, good_feature_names = self.filter_out_number(X, feature_names)

			'''
				P format [(label1, prob1),...]
			'''
			P = []
			log = dict()
			log['content'] = raw
			for label in labels:
				prob = np.float64(self.count_dict[label]['#PROB'])
				for i in range(len(good_feature_names)):
					w = good_feature_names[i]
					freq = X[0,i]
					if w in self.label_words[label]:
						prob *= np.power(self.count_dict[label][w],freq)

					else:
						prob *= self.count_dict[label]['#NOTEXIST']
				P.append((label, prob))
				log[label] = prob

			P = sorted(P, key = lambda x: x[1], reverse = True)
			if is_show :
				print('\n [INFO] ', raw[:20], ' probs : ', P)
			predicted_cls.append(P[0][0])

			log['predicted_category'] = P[0][0]

			info = info.append(log, ignore_index=True)

		if not is_show:
			print(' Done! More detail in prediction.csv')

		return predicted_cls, info

	def debug(self):
		data = self.data

		labels = data['#CAT'].unique()
		features = data.columns.values

		for label in labels :
			print('[DEBUG]', label, self.count_dict[label]['#COUNT'])

		pprint(self.count_dict)
