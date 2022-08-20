# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 14:37
# @Author  : Leesure
# @File    : visualize.py
# @Software: PyCharm
# demo.py

import matplotlib.pyplot as plt
import numpy as np


def heatmap(data, Y_labels=None, X_labels=None,
            cbar_kw: dict = None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    Y_labels
        A list or array of length M with the labels for the rows.
    X_labels
        A list or array of length N with the labels for the columns.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    fig, ax = plt.subplots()
    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    if X_labels is None or Y_labels is None:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    else:
        ax.set_xticks(np.arange(data.shape[1]), labels=X_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=Y_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)
    #
    # ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    # ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # switch backend
    plt.switch_backend('agg')

    fig, ax = plt.subplots()
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    data = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    im, cbar = heatmap(data, vegetables, farmers, ax=ax, cbarlabel="harvest [t/year]", cmap="YlGn")
    # im = ax.imshow(data)
    # plt.axis('scaled')
    fig.tight_layout()
    plt.show()
