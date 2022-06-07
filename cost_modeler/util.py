import numpy as np
import matplotlib.pyplot as plt


def plot_cdf(values, ax=None, label=None) -> plt.Axes:
    sorted_data = np.sort(values)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    if ax is not None:
        ax.plot(sorted_data, yvals, label=label)
    else:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(sorted_data, yvals, label=label, linewidth=3)
        ax.set_xlabel("Error (%)")
        ax.set_ylabel("CDF")
    return ax
