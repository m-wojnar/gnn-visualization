import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data


def plot(X: np.ndarray, y: np.ndarray, title: str = None, path: str = None) -> None:
    plt.figure(figsize=(6, 6))

    for y_val in np.unique(y):
        plt.scatter(X[y == y_val, 0], X[y == y_val, 1], s=5, label=y_val)

    plt.legend(loc='upper left')

    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    # plt.show()


def generate_plot(model: nn.Module, graph: Data, title: str = None, path: str = None) -> None:
    out = model.cpu()(graph.x, graph.edge_index, graph.edge_attr).detach().cpu().numpy()
    plot(out, graph.y.detach().cpu().numpy(), title, path)
