from argparse import ArgumentParser
import os
import pickle
from typing import List, Tuple

import faiss
import lz4.frame
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from torch import Tensor, device
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import trange

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

ALL_DATASETS = {**TRAIN_DATASETS, **TEST_DATASETS}

Graph = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
TorchGraph = Tuple[Tensor, Tensor, Tensor, Tensor, int]


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
        Number of nodes in one graph.
    n_graphs : int
        Number of subgraphs to generate.
    """

    def __init__(
            self,
            dataset_id: int,
            nn: int,
            rn: int,
            metric: str = 'binary',
            examples: int = 4000,
            n_graphs: int = 50
    ) -> None:
        self.dataset_id = dataset_id
        self.nn = nn
        self.rn = rn
        self.metric = metric
        self.examples = examples
        self.n_graphs = n_graphs

        self.X = None
        self.y = None
        self.index_flat = None

        self.graphs = []

    def run(self) -> None:
        with Timer('Training index...'):
            self.train()

        for _ in trange(1, self.n_graphs + 1):
            self.search(self.examples)

        with Timer(f'Searching all nodes...'):
            self.search(None)

    def train(self) -> None:
        self.X, self.y = fetch_openml(data_id=self.dataset_id, cache=True, return_X_y=True, as_frame=False, parser='auto')

        self.X = self.X.astype(np.float32, order='C')
        self.y = LabelEncoder().fit_transform(self.y)
        n, m = self.X.shape

        if self.metric == 'cosine':
            norm = np.linalg.norm(self.X, axis=1).reshape(-1, 1)
            self.X /= norm
            
        quantizer = faiss.IndexFlatIP(m) if self.metric == 'cosine' else faiss.IndexFlatL2(m)
        index_flat = faiss.IndexIVFFlat(quantizer, m, int(np.sqrt(n)))


        # res = faiss.StandardGpuResources()  # use a single GPU
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        assert not index_flat.is_trained
        index_flat.train(self.X)
        assert index_flat.is_trained

        index_flat.add(self.X)
        index_flat.nprobe = 10

        self.index_flat = index_flat

    def search(self, examples: int) -> None:
        n = self.X.shape[0]

        if examples is not None:
            example_idxs = np.random.choice(n, size=examples)
            X_search = self.X[example_idxs]
        else:
            X_search = self.X

        distance, indexes = self.index_flat.search(X_search, self.nn + 1)

        if self.rn > 0:
            rn_size = examples if examples is not None else n
            rn_indexes = np.random.choice(n, size=(rn_size, self.rn))
            indexes = np.concatenate([indexes, rn_indexes], axis=1)

            if self.metric == 'euclidean':
                rn_distances = np.sum((self.X[rn_indexes] - X_search[:, None]) ** 2, axis=2)
            elif self.metric == 'cosine':
                rn_distances = 1 - np.sum(self.X[rn_indexes] * X_search[:, None], axis=2)
            else:
                rn_distances = np.empty_like(rn_indexes, dtype=np.float32)

            distance = np.concatenate([distance, rn_distances], axis=1)

        # normalize distances
        norm = np.linalg.norm(self.X)
        distance /= norm

        if self.metric == 'binary':
            distance[:, :self.nn + 1] = 0.
            distance[:, self.nn + 1:] = 1.

        if examples is not None:
            idxs_all = np.unique(indexes)
            idxs_dict = {idx: i for i, idx in enumerate(idxs_all)}

            X, y = self.X[idxs_all], self.y[idxs_all]
            indexes = np.vectorize(idxs_dict.get)(indexes)
        else:
            X, y = self.X, self.y

        self.graphs.append((X, y, distance[:, 1:], indexes[:, 1:], self.nn + self.rn))

    def save(self, path: str) -> None:
        with lz4.frame.open(path, 'wb') as f:
            pickle.dump(self.graphs, f)

    @staticmethod
    def load(path: str) -> List[Graph]:
        with lz4.frame.open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_torch(path: str, dev: device) -> List[TorchGraph]:
        to_torch = lambda values: tuple(map(lambda x: torch.from_numpy(x).to(dev), values[:-1])) + (values[-1],)
        return list(map(to_torch, FaissGenerator.load(path)))

    @staticmethod
    def load_dataset(path: str, dev: device, batch_size: int = 8, shuffle: bool = True) -> Tuple[Data, DataLoader]:
        graphs = FaissGenerator.load_torch(path, dev)
        graphs = list(map(lambda x: FaissGenerator.create_graph(*x, dev), graphs))
        subgraphs, graph = graphs[:-1], graphs[-1]

        return graph, DataLoader(subgraphs, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def create_graph(X: Tensor, y: Tensor, distances: Tensor, indexes: Tensor, n_neighbours: int, dev: device) -> Data:
        row = torch.arange(indexes.shape[0]).view(-1, 1).repeat(1, n_neighbours).view(-1).to(dev)
        col = indexes.contiguous().view(-1)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = distances.contiguous().view(-1, 1)

        perm = torch.randperm(edge_index.shape[1]).to(dev)
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='mnist_784')
    args.add_argument('--nn', type=int, default=2)
    args.add_argument('--rn', type=int, default=1)
    args.add_argument('--metric', default='binary', choices=['binary', 'cosine', 'euclidean'])
    args.add_argument('--examples', type=int, default=4000)
    args.add_argument('--n_graphs', type=int, default=100)
    args.add_argument('--save_path', type=str, default=None)
    args = args.parse_args()

    if args.save_path is not None:
        save_path = args.save_path
    else:
        examples = f'{args.examples}ex' if args.examples is not None else 'full'
        filename = f'{args.n_graphs}g_{examples}_{args.metric}_{args.nn}nn_{args.rn}rn.pkl.lz4'
        save_path = os.path.join(ROOT_PATH, 'data', args.dataset, filename)
    
    generator = FaissGenerator(
        ALL_DATASETS[args.dataset], nn=args.nn, rn=args.rn, metric=args.metric,
        examples=args.examples, n_graphs=args.n_graphs
    )
    generator.run()
    generator.save(save_path)
