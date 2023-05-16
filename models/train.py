from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

from data import FaissGenerator


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 2) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))

        x = self.convs[-1](x, edge_index, edge_attr)
        return x


def train(args: Dict, loader: DataLoader) -> nn.Module:
    model = GCN(input_dim=args['input_dim'], hidden_dim=64, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.HuberLoss()

    for epoch in range(args['epochs']):
        for data in loader:
            raise NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.save(model.state_dict(), f'checkpoints/gcn_model_{epoch}.pt')

    return model


def generate_visualization(model: nn.Module, loader: DataLoader) -> None:
    raise NotImplementedError


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--data', type=str, default='../data/mnist/mnist_784_nn100_cosine.pkl.lz4')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--hidden_dim', type=int, default=64)
    args.add_argument('--lr', type=float, default=0.01)

    args = args.parse_args()
    args = vars(args)

    X, y, distances, indexes = FaissGenerator.load(args['data'])
    args['input_dim'] = X.shape[1]

    dataset = None  # TODO
    loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    model = train(args, loader)
    model.eval()
    torch.save(model.state_dict(), 'checkpoints/gcn_model_final.pt')
