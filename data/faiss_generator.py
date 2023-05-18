import pickle
from typing import Tuple

import faiss
import lz4.frame
import numpy as np
from sklearn.datasets import fetch_openml

from utils import Timer


class FaissGenerator:
    def __init__(self, dataset_name: str, nn: int, cosine_metric: bool = False) -> None:
        self.nn = nn
        self.cosine_metric = cosine_metric
        self.dataset_name = dataset_name

        self.X = None
        self.y = None
        self.distances = None
        self.indexes = None

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with Timer('Downloading dataset...'):
            self.X, self.y = fetch_openml(self.dataset_name, cache=True, return_X_y=True, as_frame=False, parser='liac-arff')
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


        # res = faiss.StandardGpuResources()  # use a single GPU
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

        return self.X, self.y, self.distances, self.indexes

    def save(self, path: str) -> None:
        with lz4.frame.open(path, 'wb') as f:
            pickle.dump((self.X, self.y, self.distances, self.indexes), f)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with lz4.frame.open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    generator = FaissGenerator('mnist_784', nn=100, cosine_metric=True)
    generator.run()
    generator.save('mnist/mnist_784_nn100_cosine.pkl.lz4')

    generator = FaissGenerator('mnist_784', nn=100, cosine_metric=False)
    generator.run()
    generator.save('mnist/mnist_784_nn100_euclidean.pkl.lz4')
