from math import floor
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os, sys
from typing import Dict, List, Any, Tuple

from src.config.structures import graph_axes, GRAPH_TYPES, SavePaths
from src.util.file_helper import create_filename, save
from src.util.plot_helper import color_dict

def idx_sizes_dict(data: Dict[Any, Any]) -> Tuple[int,Dict[Any, Any]]:
    tmp = dict()
    largest_size = 0
    for key, values in data.items():
        data_length = len(values[0])
        largest_size = data_length if data_length > largest_size else largest_size
        index_size = floor(0.067 * data_length) if floor(0.067 * data_length) >= 1 else 1 
        tmp[key] = index_size
    return largest_size, tmp

def animate(i: int, ax: Axes, idx_sizes: Dict[Any, Any], colors: Dict[str, str], graph_type: str, data: Dict[Any, Any]):
    try:
        lines = None
        if graph_type == GRAPH_TYPES.LINE.value:
            lines = [ax.plot(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == GRAPH_TYPES.SCATTER.value:
            lines = [ax.scatter(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == GRAPH_TYPES.LINE3D.value:
            lines = [ax.plot(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], val[2][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == GRAPH_TYPES.SCATTER3D.value:
            lines = [ax.scatter(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], val[2][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in data.items()]  
        return lines,
    except Exception as e:
        print(e)

def animate_graph(fig: Figure, ax: Axes, title: str, graph_type: str, data: Dict[str, Any], save_paths: SavePaths) -> None:
    colors = color_dict(list(data.keys()))
    for key, val in data.items():
        if graph_type == GRAPH_TYPES.LINE.value:
            ax.plot(val[0][0], val[1][0], c=colors[key], label=key)
        elif graph_type == GRAPH_TYPES.LINE3D.value:
            ax.plot(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        elif graph_type == GRAPH_TYPES.SCATTER.value:
            ax.scatter(val[0][0], val[1][0], c=colors[key], label=key)
        elif graph_type == GRAPH_TYPES.SCATTER3D.value:
            ax.scatter(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
    plt.legend()
    
    data_length, idx_sizes = idx_sizes_dict(data)
    index_size = floor(0.05 * data_length) if floor(0.05 * data_length) >= 1 else 1 
    frame_size = data_length//index_size

    anim = animation.FuncAnimation(fig, animate, frames=frame_size,fargs=(ax, idx_sizes, colors, graph_type, data), interval=1, repeat=False) # type: ignore
    save(title, save_paths, anim)