from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


import ipywidgets as widgets
from IPython.display import display



class EmbeddingsScatterAnimation:
    def __init__(self, H: np.array, C=None, figsize=None, xlim=None, ylim=None, interval=50, titleFormat=None):
        """H - iterable of embeddings (n x 2) => np.array of shape (embeddings_no, samples_no, 2)
        """
        self.figsize = figsize
        self.xlim = xlim
        self.ylim = ylim

        self.H = H
        self.C = C
        self.titleFormat = titleFormat

        self.intervalSlider = widgets.IntSlider(
            value=100,
            min=1,
            max=1000,
            step=1,
            description='Test:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        # TODO Slider
        self.intervalSlider.observe(self.on_interval_change)
        # display(self.intervalSlider)

        self.setup(interval)

    def on_interval_change(self, change):
        nv = change['new']
        if type(nv) == int:
            self.anim.pause()
            del self.anim
            interval = nv
            # print(change)
            self.anim = animation.FuncAnimation(self.fig, self.animfunc, interval=interval, frames=self.H.shape[0])


    def animfunc(self, i):
        C = self.C if self.C is not None else None
        self.ax.clear()
        if self.xlim is not None:
            self.ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(*self.ylim)
        self.plot = sns.scatterplot(x=self.H[i, :, 0], y=self.H[i, :, 1], hue=C, ax=self.ax)
        # ftm = self.titleFormat if self.titleFormat is not None else ""
        self.plot.set(title=f"iter {i}")

    def setup(self, interval=50):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.subplot()
       
        self.sctr  = sns.scatterplot(x=self.H[0, :, 0], y=self.H[0, :, 1], ax=self.ax)
        self.plot = None
        self.runBtn = widgets.Button(description="Run anim")
        self.stopBtn = widgets.Button(description="Stop anim")
        self.closeBtn = widgets.Button(description="Close anim")

        display(self.runBtn)
        display(self.stopBtn)
        # display(self.closeBtn)

        self.anim = animation.FuncAnimation(self.fig, self.animfunc, interval=interval, frames=self.H.shape[0])
        # print("anim created")
        self.runBtn.on_click(lambda _:self.anim.resume())
        self.stopBtn.on_click(lambda _:self.anim.pause())
        self.closeBtn.on_click(lambda _:self.close())

    def close(self):
        """
        Destroy animation
        """
        plt.close(self.fig)
        del self.fig
        del self.anim