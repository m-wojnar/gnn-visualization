# gnn-visualization

## Installation

In order to install the package and its dependencies, run the following commands:

```bash
# clone the repository
git clone git@github.com:m-wojnar/gnn-visualization.git
cd gnn-visualization

# create a virtual environment and install the dependencies
python3 -m venv venv
pip install -r requirements.txt
```

**Note:** you may need to install `faiss` package manually.

## Usage

### Faiss index creation

By default, the `FaissGenerator` class uses the `openml` package to download the datasets. You can specify the dataset
you want to use by passing its id to the `FaissGenerator` constructor. The `FaissGenerator` class also takes the
`nn` parameter, which specifies the number nearest neighbors, and `cosine_metric`, which specifies the metric used to
compute the nearest neighbors.

```python
from gnn_visualization import FaissGenerator

generator = FaissGenerator('mnist_784', nn=100, cosine_metric=True)
X, distances, indexes = generator.run()
```

You can save the results with pickle and lz4 compression:

```python
generator.save('mnist_784_nn100_cosine.pkl.lz4')
```

and load them later:

```python
X, distances, indexes = FaissGenerator.load('mnist_784_nn100_cosine.pkl.lz4')
```