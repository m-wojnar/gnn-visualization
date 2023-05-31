from argparse import ArgumentParser
import os
import pickle
from typing import Tuple

import faiss
import lz4.frame
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from torch import Tensor, device
from torch_geometric.data import Data

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
    """
    Generates a nearest neighbor graph using the Faiss library. The graph can be saved in a pickle file.

    Parameters
    ----------
    dataset_id : int
        Dataset id from OpenML.
    nn : int
        Number of nearest neighbors.
    rn : int
        Number of random neighbors.
    metric : str
        Metric used to compute distances. Can be 'binary', 'cosine' or 'euclidean'.
    examples : int
        Number of examples to use from the dataset. If None, all examples are used.
    """

    def __init__(
            self,
            dataset_id: int,
            nn: int,
            rn: int,
            metric: str = 'binary',
            examples: int = None
    ) -> None:
        self.dataset_id = dataset_id
        self.nn = nn
        self.rn = rn
        self.metric = metric
        self.examples = examples

        self.X = None
        self.y = None
        self.distances = None
        self.indexes = None

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with Timer('Downloading dataset...'):
            self.X, self.y = fetch_openml(data_id=self.dataset_id, cache=True, return_X_y=True, as_frame=False, parser='auto')

        if self.examples is not None:
            idxs = np.random.choice(self.X.shape[0], self.examples)
            self.X, self.y = self.X[idxs], self.y[idxs]

        self.X = self.X.astype(np.float32)
        self.y = LabelEncoder().fit_transform(self.y)
        n, m = self.X.shape

        if self.metric == 'cosine':
            norm = np.linalg.norm(self.X, axis=1).reshape(-1, 1)
            self.X /= norm
            
        quantizer = faiss.IndexFlatIP(m) if self.metric == 'cosine' else faiss.IndexFlatL2(m)
        index_flat = faiss.IndexIVFFlat(quantizer, m, int(np.sqrt(n)))

        with Timer('Building index...'):
            # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            assert not index_flat.is_trained
            index_flat.train(self.X)
            assert index_flat.is_trained

            index_flat.add(self.X)

        with Timer('Searching...'):
            index_flat.nprobe = 10
            self.distances, self.indexes = index_flat.search(self.X, self.nn + 1)
            self.distances, self.indexes = self.distances[:, 1:], self.indexes[:, 1:]

        if self.rn > 0:
            rn_indexes = np.random.choice(n, size=(n, self.rn))
            self.indexes = np.concatenate([self.indexes, rn_indexes], axis=1)

            if self.metric == 'euclidean':
                rn_distances = np.sum((self.X[rn_indexes] - self.X[:, None]) ** 2, axis=2)
            elif self.metric == 'cosine':
                rn_distances = 1 - np.sum(self.X[rn_indexes] * self.X[:, None], axis=2)
            else:
                rn_distances = np.ones_like(rn_indexes, dtype=np.float32)

            self.distances = np.concatenate([self.distances, rn_distances], axis=1)

        # normalize distances
        norm = np.linalg.norm(self.X)
        self.distances /= norm

        if self.metric == 'binary':
            self.distances[:, :self.nn] = 0.
            self.distances[:, self.nn:] = 1.

        return self.X, self.y, self.distances, self.indexes, self.nn + self.rn

    def save(self, path: str) -> None:
        with lz4.frame.open(path, 'wb') as f:
            pickle.dump((self.X, self.y, self.distances, self.indexes, self.nn + self.rn), f)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with lz4.frame.open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_torch(path: str, dev: device) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        values = FaissGenerator.load(path)
        return tuple(map(lambda x: torch.from_numpy(x).to(dev), values[:-1])) + (values[-1],)

    @staticmethod
    def load_graph(path: str, dev: device) -> Data:
        return FaissGenerator.create_graph(*FaissGenerator.load_torch(path, dev))

    @staticmethod
    def create_graph(X: Tensor, y: Tensor, distances: Tensor, indexes: Tensor, n_neighbours: int) -> Data:
        row = torch.arange(X.shape[0]).view(-1, 1).repeat(1, n_neighbours).view(-1)
        col = indexes.contiguous().view(-1)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = distances.contiguous().view(-1, 1)

        return Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='mnist_784')
    args.add_argument('--nn', type=int, default=2)
    args.add_argument('--rn', type=int, default=1)
    args.add_argument('--metric', default='binary', choices=['binary', 'cosine', 'euclidean'])
    args.add_argument('--examples', type=int, default=None)
    args.add_argument('--save_path', type=str, default=None)
    args = args.parse_args()

    if args.save_path is not None:
        save_path = args.save_path
    else:
        examples = f'ex{args.examples}' if args.examples is not None else 'full'
        save_path = os.path.join(ROOT_PATH, 'data', args.dataset, f'{args.metric}_{examples}_nn{args.nn}_rn{args.rn}.pkl.lz4')
    
    generator = FaissGenerator(TRAIN_DATASETS[args.dataset], nn=args.nn, rn=args.rn, metric=args.metric, examples=args.examples)
    generator.run()
    generator.save(save_path)
