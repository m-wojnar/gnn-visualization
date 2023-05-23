from argparse import ArgumentParser
import os
import pickle
from typing import Tuple

import faiss
import lz4.frame
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from utils import Timer, ROOT_PATH

TRAIN_DATASETS = {
    'mnist_784':        554,    # 10 classes digits
    'Kuzushiji-MNIST':  41982,  # 10 classes Kuzushiji (cursive Japanese)
    'SignMNIST':        45082,  # 24 classes A-Z letters
    'gina_prior':       1042    # 2 classes (odd and even) MNIST digits
}

TEST_DATASETS = {
    'Fashion-MNIST':    40996,  # 10 classes Zalando clothes
}


class FaissGenerator:
    def __init__(self, dataset_id: int, nn: int, cosine_metric: bool = False, limit_examples: int = None) -> None:
        self.nn = nn
        self.cosine_metric = cosine_metric
        self.dataset_id = dataset_id
        self.limit_examples = limit_examples

        self.X = None
        self.y = None
        self.distances = None
        self.indexes = None

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with Timer('Downloading dataset...'):
            self.X, self.y = fetch_openml(data_id=self.dataset_id, cache=True, return_X_y=True, as_frame=False, parser='auto')

        if self.limit_examples is not None:
            idxs = np.random.choice(self.X.shape[0], self.limit_examples)
            self.X, self.y = self.X[idxs], self.y[idxs]

        self.X = self.X.astype(np.float32)
        self.y = LabelEncoder().fit_transform(self.y)
        n, m = self.X.shape

        if self.cosine_metric:
            norm = np.linalg.norm(self.X, axis=1).reshape(-1, 1)
            self.X /= norm
            
        if self.cosine_metric:
            quantizer = faiss.IndexFlatIP(m)
            index_flat = faiss.IndexIVFFlat(quantizer, m, int(np.sqrt(n)))
            index_flat.nprobe = 10
        else:
            quantizer = faiss.IndexFlatL2(m)
            index_flat = faiss.IndexIVFFlat(quantizer, m, int(np.sqrt(n)))

        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        assert not index_flat.is_trained
        index_flat.train(self.X)
        assert index_flat.is_trained

        index_flat.add(self.X)

        with Timer('Searching...'):
            index_flat.nprobe = 10
            self.distances, self.indexes = index_flat.search(self.X, self.nn + 1)

        # normalize distances
        norm = np.linalg.norm(self.X)
        self.distances /= norm

        return self.X, self.y, self.distances, self.indexes, self.nn

    def save(self, path: str) -> None:
        with lz4.frame.open(path, 'wb') as f:
            pickle.dump((self.X, self.y, self.distances, self.indexes, self.nn), f)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with lz4.frame.open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='mnist_784')
    args.add_argument('--save_path', type=str, default=f'{ROOT_PATH}/data/mnist_784/dataset_nn100.pkl.lz4')
    args.add_argument('--cosine', default=False, action='store_true')
    args.add_argument('--limit_examples', type=int, default=4000)
    args.add_argument('--nn', type=int, default=100)
    args = args.parse_args()
    
    generator = FaissGenerator(TRAIN_DATASETS[args.dataset], nn=args.nn, cosine_metric=args.cosine, limit_examples=args.limit_examples)
    generator.run()
    generator.save(args.save_path)
