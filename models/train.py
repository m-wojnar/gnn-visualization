from argparse import ArgumentParser
from functools import reduce

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, CGConv

from data import FaissGenerator
from utils import ROOT_PATH


class VisGNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            edge_dim: int = 1,
            output_dim: int = 2,
            aggr: str = 'mean'
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [CGConv(channels=hidden_dim, dim=edge_dim, aggr=aggr) for _ in range(num_layers - 2)] +
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.layers[0](x, edge_index).relu()
        x = reduce(lambda x, layer: layer(x, edge_index, edge_attr).relu(), self.layers[1:-1], x)
        x = self.layers[-1](x, edge_index)
        return x


def train(model: nn.Module, graph: Data, epochs: int, lr: float) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        distances = F.pairwise_distance(out[graph.edge_index[0]], out[graph.edge_index[1]]).view(-1, 1)
        loss = criterion(distances, graph.edge_attr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, loss: {loss.item()}')
        torch.save(model.state_dict(), f'{ROOT_PATH}/models/checkpoints/vis_gnn_model_{epoch}.pt')

        if epoch % 10 == 0:
            model.eval()
            generate_visualization(model, graph, f'{ROOT_PATH}/models/checkpoints/vis_gnn_{epoch}.pdf')
            model.train()

    return model


def generate_visualization(model: nn.Module, graph: Data, path) -> None:
    out = model(graph.x, graph.edge_index, graph.edge_attr).detach().cpu().numpy()
    plt.figure(figsize=(6, 6))

    for y in graph.y.unique():
        plt.scatter(out[graph.y == y, 0], out[graph.y == y, 1], s=5, label=y.item())

    plt.title('VisGNN')
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.show()


def create_graph(X: Tensor, y: Tensor, distances: Tensor, indexes: Tensor, nn_val: int) -> Data:
    row = torch.arange(X.shape[0]).view(-1, 1).repeat(1, nn_val).view(-1)
    col = indexes.contiguous().view(-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = distances.contiguous().view(-1, 1)

    return Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data', type=str, default=f'{ROOT_PATH}/data/mnist_784/dataset_nn100.pkl.lz4')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--hidden_dim', type=int, default=64)
    args.add_argument('--num_layers', type=int, default=5)
    args.add_argument('--lr', type=float, default=0.001)
    args = args.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y, distances, indexes, nn_val = FaissGenerator.load(args.data)
    X, y, distances, indexes = tuple(map(lambda x: torch.from_numpy(x).to(device), (X, y, distances[:, 1:], indexes[:, 1:])))
    graph = create_graph(X, y, distances, indexes, nn_val)

    model = VisGNN(input_dim=X.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    model = train(model, graph, args.epochs, args.lr)
    model.eval()

    torch.save(model.state_dict(), f'{ROOT_PATH}/models/checkpoints/vis_gnn_model_final.pt')
    generate_visualization(model, graph, f'{ROOT_PATH}/models/checkpoints/vis_gnn_final.pdf')
