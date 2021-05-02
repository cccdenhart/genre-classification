import numpy as np
from src import prepare_data, build_features, split_data, CACHE_DIR
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
from typing import Tuple
import os


def train_rf(X: np.array, y: np.array) -> RandomForestClassifier:
	'''Train a random forest classifier on the data.'''
	rf = RandomForestClassifier().fit(X, y)
	return rf


def train_svc(X: np.array, y: np.array) -> svm.SVC:
	'''Train a support vector classifier on the data.'''
	svc = svm.SVC(decision_function_shape='ovo', kernel='linear').fit(X, y)
	return svc


def train_baseline() -> Tuple[RandomForestClassifier, svm.SVC]:
	'''Train baseline models.'''
	print('Loading data .....')
	X, y, y_classes = prepare_data()
	derived_X, _ = build_features(X)
	X_train, y_train, X_test, y_test = split_data(derived_X, y, row_cache_name='baseline')

	print('Training random forest .....')
	rf = train_rf(X_train, y_train)
	pickle.dump(rf, open(os.path.join(CACHE_DIR, 'rf.pk'), 'wb'))

	print('Training support vector classifier .....')
	svc = train_svc(X_train, y_train)
	pickle.dump(svc, open(os.path.join(CACHE_DIR, 'svc.pk'), 'wb'))

	return rf, svc


if __name__ == '__main__':
	train_baseline()
