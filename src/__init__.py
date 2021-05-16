import numpy as np
from typing import List, Callable, Tuple, Optional, Dict
import librosa
import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.genre_embedder import GenreEmbedder

# project constants
DATA_DIR = '/home/denhart.c/data'
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CACHE_DIR = os.path.join(PROJECT_DIR, 'cache')
SAMPLE_RATE = 22050

if not os.path.exists(CACHE_DIR):
	os.mkdir(CACHE_DIR)


def audio_to_sequence(
	x: np.ndarray,
	use_mel: bool = True,
	to_db: bool = False,
	time_seq: bool = False
) -> np.ndarray:
	'''Convert an audio wave to a sequence representation.'''
	if use_mel:
		return librosa.features.mel_spectogram(x)
	amplitudes = abs(librosa.stft(x))
	if to_db:
		amplitudes = librosa.amplitude_to_db(time_amplitudes)
	if time_seq:
		amplitudes = amplitudes.reshape(amplitudes.shape[1],
		amplitudes.shape[0])
	return amplitudes


def sequence_to_audio(X: np.ndarray, from_db: bool = False) -> np.ndarray:
	'''Convert sequence represented audio back to raw waveform.'''
	values = librosa.db_to_amplitude(X) if from_db else X
	return librosa.istft(values)


def embed_X(X: List[np.ndarray], embed_fn: Callable[[np.ndarray], np.ndarray]) -> List[np.ndarray]:
	'''Embed raw audio data using the given function.'''
	embeddings = [embed_fn(x) for x in X]
	return embeddings


def read_data(subset_pct: float = 1.0) -> Tuple[List[str], List[str]]:
	'''Return audio filepaths and associated genre labels'''
	# extract all genre names
	genre_dir = os.path.join(DATA_DIR, 'genres')
	genre_names = [fname for fname in os.listdir(genre_dir)
				   if os.path.isdir(os.path.join(genre_dir, fname))]

	# collect filepaths along with their associated genres
	filepaths = []
	labels = []
	for genre in genre_names:
		genre_files = os.listdir(os.path.join(DATA_DIR, 'genres', genre))
		genre_paths = [os.path.join(DATA_DIR, 'genres', genre, fname) for fname in genre_files]
		filepaths += genre_paths
		labels += ([genre] * len(genre_files))

	# select random subset of files
	if subset_pct < 1.0:
		n_total_files = len(filepaths)
		n_subset_files = int(subset_pct * n_total_files)
		random_rows = np.random.choice(range(n_total_files), n_subset_files, replace=False)
		filepaths_subset = [filepaths[i] for i in random_rows]
		labels_subset = [labels[i] for i in random_rows]

		return filepaths_subset, labels_subset

	return filepaths, labels


def prepare_data(subset_pct: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
	'''Prepare data for model training.'''
	filepaths, labels = read_data(subset_pct)

	y_encoder = LabelEncoder()
	y = y_encoder.fit_transform(labels)
	y_classes = y_encoder.classes_

	X = [librosa.load(fpath)[0] for fpath in filepaths]

	return X, y, y_classes


def split_data(
	X: np.ndarray,
	y: np.ndarray,
	train_rows: List[int] = None,
	val_rows: List[int] = None,
	use_val: bool = False,
	row_cache_name: Optional[str] = None
) -> List[np.ndarray]:
	'''Split the data into train, (val), and test sets.'''
	n_total_rows = X.shape[0]
	train_split, val_split, test_split = 0.6, 0.1, 0.3

	if not train_rows:
		train_rows = np.random.choice(n_total_rows, int(train_split * n_total_rows), replace=False)
		if row_cache_name:
			pd.DataFrame(train_rows).to_csv(os.path.join(CACHE_DIR, f'{row_cache_name}_train_rows.csv'), index=False, header=False)

	if use_val:
		avail_rows = [i for i in range(n_total_rows) if i not in train_rows]

		if not val_rows:
			val_rows = np.random.choice(avail_rows, int(val_split * n_total_rows), replace=False)
			if row_cache_name:
				pd.DataFrame(val_rows).to_csv(os.path.join(CACHE_DIR, f'{row_cache_name}_val_rows.csv'), index=False, header=False)


		test_rows = [i for i in range(n_total_rows)
					 if i not in train_rows and i not in val_rows]
	else:
		test_rows = [i for i in range(n_total_rows) if i not in train_rows]


	out_data = [
        X[train_rows],
        y[train_rows],
        X[test_rows],
        y[test_rows]
    ]

	if use_val:
		out_data += [X[val_rows], y[val_rows]]

	return out_data


def get_feature_names() -> List[str]:
	'''Build a hardcoded list of derived feature names.'''
	derived_feature_names = [
		'chroma_mean',
		'chroma_var',
		'spec_cent_mean',
		'spec_cent_var',
		'spec_band_mean',
		'spec_band_var',
		'spec_roll_mean',
		'spec_roll_var',
		'zcr_mean',
		'zcr_var',
		'tempogram_mean',
		'tempogram_var',
	]
	derived_feature_names += ['mfcc_mean_' + str(i) for i in range(20)] \
						  + ['mfcc_var_' + str(i) for i in range(20)]
	return derived_feature_names


def derived_features(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
	'''Build derived features using librosa.'''
	chroma_stft = librosa.feature.chroma_stft(x, sr)
	spec_cent = librosa.feature.spectral_centroid(x, sr)
	spec_bw = librosa.feature.spectral_bandwidth(x, sr)
	rolloff = librosa.feature.spectral_rolloff(x, sr)
	zcr = librosa.feature.zero_crossing_rate(x)
	tempogram = librosa.feature.tempogram(x, sr)
	mfcc = librosa.feature.mfcc(x, sr)
	mfcc_means = np.stack([np.mean(e) for e in mfcc])
	mfcc_vars = np.stack([np.std(e) for e in mfcc])
	feature_stack = np.stack([
		np.mean(chroma_stft),
		np.std(chroma_stft),
		np.mean(spec_cent),
		np.std(spec_cent),
		np.mean(spec_bw),
		np.std(spec_bw),
		np.mean(rolloff),
		np.std(rolloff),
		np.mean(zcr),
		np.std(zcr),
		np.mean(tempogram),
		np.std(tempogram)
	])
	full_stack = np.concatenate([feature_stack, mfcc_means, mfcc_vars])
	return full_stack


def build_features(X: np.array, y: np.array, use_cache: bool = True, cache: bool = True) -> Tuple[np.array, torch.tensor]:
	'''Build derived and sequential features.'''
	if use_cache:
		derived_X = pd.read_csv(os.path.join(CACHE_DIR, 'derived_df.csv')).values
		seq_X = torch.load(os.path.join(CACHE_DIR, 'sequence_X.pt'))
		y = pd.read_csv(os.path.join(CACHE_DIR, 'y.csv'), header=None).squeeze().values
	else:
		# load derived features
		emb_X = embed_X(X, lambda x: derived_features(x))
		derived_X = torch.tensor(np.stack([x for x in emb_X])).float()

		# load sequential features
		emb_X = embed_X(X, audio_to_sequence)
		min_len = min([x.shape[1] for x in emb_X])
		seq_X = torch.tensor(np.stack([x[:, :min_len] for x in emb_X])).float()

		if cache:
			derived_df = pd.DataFrame(derived_X.numpy())
			derived_df.columns = get_feature_names()
			derived_df.to_csv(os.path.join(CACHE_DIR, 'derived_df.csv'), index=False)

			torch.save(seq_X, os.path.join(CACHE_DIR, 'sequence_X.pt'))

			pd.DataFrame(y).to_csv(os.path.join(CACHE_DIR, 'y.csv'), index=False)

	return derived_X, seq_X, y


def assign_embeddings(X: torch.tensor, y: torch.tensor, embedder: GenreEmbedder) -> Dict[int, torch.tensor]:
    genre_embeddings = {}
    for genre in set(y.tolist()):
        y_idx = torch.where(y == genre)[0]
        X_y = X[y_idx]
        _, embs = embedder(X_y)
        mean_embs = torch.mean(embs, dim=0)
        genre_embeddings[genre] = mean_embs
    return genre_embeddings
