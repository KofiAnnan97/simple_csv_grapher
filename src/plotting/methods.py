from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
import numpy as np
import os, sys
from typing import Dict, Any, List, Tuple

from src.config.structures import Parameters, graph_axes, GRAPH_TYPES, SavePaths
from src.plotting.animate import animate_graph
from src.util.plot_helper import color_dict, get_limits
from src.util.file_helper import save

def second_axis(p: Parameters, sp: SavePaths) -> None:
    fig, ax1 = plt.subplots()
    static_axis = ''
    two_axes = list()
    ax2 = ax1.twiny()
    if p.graph_type in [GRAPH_TYPES.LINE_XX.value, GRAPH_TYPES.SCATTER_XX.value]:
        static_axis = p.labels[1]
        two_axes = p.labels[0].split(",")[:2]
    elif p.graph_type in [GRAPH_TYPES.LINE_YY.value, GRAPH_TYPES.SCATTER_YY.value]:
        static_axis = p.labels[0]
        two_axes = p.labels[1].split(",")[:2]
        ax2 = ax1.twinx()
    ax1.set_xlabel(static_axis)
    colors = color_dict(two_axes)
    ax1.set_ylabel(two_axes[0], color=colors[two_axes[0]])
    ax2.set_ylabel(two_axes[1], color=colors[two_axes[1]])
    plt.title(p.title)

    plots = []
    for key, val in p.data.items():
        points = val[1]
        plot = None
        if val[0] == 1:
            if p.graph_type in [GRAPH_TYPES.LINE_XX.value, GRAPH_TYPES.LINE_YY.value]:
                plot = ax1.plot(points[0], points[1], label=key, color=colors[two_axes[0]])
            elif p.graph_type in [GRAPH_TYPES.SCATTER_XX.value, GRAPH_TYPES.SCATTER_YY.value]:
                plot = ax1.scatter(points[0], points[1], label=key, color=colors[two_axes[0]])
        elif val[0] == 2:
            if p.graph_type in [GRAPH_TYPES.LINE_XX.value, GRAPH_TYPES.LINE_YY.value]:
                plot = ax2.plot(points[0], points[1], label=key, color=colors[two_axes[1]])
            elif p.graph_type in [GRAPH_TYPES.SCATTER_XX.value, GRAPH_TYPES.SCATTER_YY.value]:
                plot = ax2.scatter(points[0], points[1], label=key, color=colors[two_axes[1]])
        if plot is not None:
            if p.graph_type in [GRAPH_TYPES.LINE_XX.value, GRAPH_TYPES.LINE_YY.value]:
                plots += plot # type: ignore
            elif p.graph_type in [GRAPH_TYPES.SCATTER_XX.value, GRAPH_TYPES.SCATTER_YY.value]:
                plots.append(plot)
    set_labels = [plot.get_label() for plot in plots]
    plt.legend(plots, set_labels,loc='upper right')
    if p.save_file:
        save(p.title, sp)
    else:
        plt.show()

def multi_line(p: Parameters, sp: SavePaths) -> None:
    num_of_axes = graph_axes[p.graph_type]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if num_of_axes == 3:
        ax = plt.axes(projection='3d')
    plt.title(p.title)

    limits = get_limits(p.data)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=p.labels[0], ylabel=p.labels[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=p.labels[0], ylabel=p.labels[1], zlabel=p.labels[2])

    if p.is_animated:
        animate_graph(fig=fig,ax=ax,title=p.title,graph_type=p.graph_type,data=p.data, save_paths=sp)
    else:
        for key, val in p.data.items():
            if num_of_axes == 2:
                ax.plot(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if p.save_file:
            save(p.title, sp)
        else:
            plt.show()

def multi_scatter(p: Parameters, sp: SavePaths) -> None:
    num_of_axes = graph_axes[p.graph_type]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   
    if num_of_axes == 3:
        ax = plt.axes(projection='3d')
        plt.title(p.title)

        limits = get_limits(p.data)
        if num_of_axes == 2:
            ax.set(xlim=limits[0], ylim=limits[1], xlabel=p.labels[0], ylabel=p.labels[1])
        elif num_of_axes == 3:
            ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=p.labels[0], ylabel=p.labels[1], zlabel=p.labels[2])

        if p.is_animated:
            animate_graph(fig=fig,ax=ax,title=p.title,graph_type=p.graph_type,data=p.data, save_paths=sp)
        else:
            for key, val in p.data.items():
                if num_of_axes == 2:
                    ax.scatter(val[0], val[1], label=key)
                elif num_of_axes == 3:
                    ax.scatter(val[0], val[1], val[2], label=key)
            plt.legend()
            if p.save_file:
                save(p.title, sp)
            else:
                plt.show()

def multi_scatter_histogram(p: Parameters, sp: SavePaths) -> None:
    print("This figure may take a while to complete based on the size of data.")
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = ax.inset_axes((0.0, 1.05, 1.0, 0.25), sharex=ax)
    ax_histy = ax.inset_axes((1.05, 0.0, 0.25, 1.0), sharey=ax)
    ax.set(xlabel=p.labels[0], ylabel=p.labels[1])
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    fig.suptitle(p.title)

    limits = get_limits(p.data)
    xmin, xmax = limits[0]
    ymin, ymax = limits[1]
    x_binwidth = (xmax - xmin)/20.0
    y_binwidth = (ymax - ymin)/20.0

    x_bins = np.arange(xmin, xmax + x_binwidth, x_binwidth)
    y_bins = np.arange(ymin, ymax + y_binwidth, y_binwidth)

    for key, val in p.data.items():
        ax.scatter(val[0], val[1], label=key)
        ax_histx.hist(val[0], bins=x_bins)
        ax_histy.hist(val[1], bins=y_bins, orientation='horizontal')

    plt.legend()
    if p.save_file:
        save(p.title, sp)
    else:
        plt.show()

def multi_histogram(p: Parameters, sp: SavePaths) -> None:
    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.suptitle(p.title)
    for key, val in p.data.items():
        axs[0].hist(val[0], bins=n_bins, label=key)
        axs[1].hist(val[1], bins=n_bins, label=key)

    axs[0].set_xlabel(p.labels[0])
    axs[1].set_xlabel(p.labels[1])
    axs[0].autoscale(enable=True, axis='both')
    axs[1].autoscale(enable=True, axis='both') 
    plt.legend()  

    if p.save_file:
        save(p.title, sp)
    else:
        plt.show()

def multi_stem(p: Parameters, sp: SavePaths) -> None:
    fig, ax = plt.subplots()
    plt.title(p.title)

    ax.set(xlabel=p.labels[0], ylabel=p.labels[1])
    ax.autoscale(enable=True, axis='both')

    colors = color_dict(list(p.data.keys()))
    for key, val in p.data.items():
        ax.stem(val[0], val[1], label=key, linefmt=colors[key], markerfmt='D')
    plt.legend()

    if p.save_file:
        save(p.title, sp)
    else:
        plt.show()

def position_perf(p: Parameters, sp: SavePaths) -> None:
    num_of_plots = graph_axes[p.graph_type]
    fig, axs = plt.subplots(num_of_plots, sharex=True)
    plt.suptitle(p.title)
    ylabels = p.labels[1].split(",")

    limits = get_limits(p.data)
    print(ylabels, limits)
    axs[0].set(xlim=limits[0], ylim=limits[1], ylabel=ylabels[0])
    if num_of_plots == 2:
        axs[1].set(xlim=limits[0], ylim=limits[2], xlabel=p.labels[0], ylabel=ylabels[1])
    elif num_of_plots == 3:
        axs[1].set(xlim=limits[0], ylim=limits[2], ylabel=ylabels[1])
        axs[2].set(xlim=limits[0], ylim=limits[3], xlabel=p.labels[0], ylabel=ylabels[2])
    
    for key, val in p.data.items():
        axs[0].plot(val[0], val[1], label=key)
        axs[1].plot(val[0], val[2], label=key)
        if num_of_plots == 3:
            axs[2].plot(val[0], val[3], label=key)
    
    plt.legend()
    if p.save_file:
        save(p.title, sp)
    else:
        plt.show()
