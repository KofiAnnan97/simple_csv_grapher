from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
import numpy as np
import os, sys
from typing import Dict, Any, List, Tuple

from src.util.parsers import CSVParser
from src.config.structures import Parameters, graph_axes, GRAPH_TYPES, SavePaths
from src.util.plot_helper import color_dict

def update(i: int, ax: Axes, metadata: Dict[Any, Any], colors: Dict[Any, Any], title: str, labels: List[str], graph_type: str) -> List[Any]:
    plt.title(title)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    lines = []
    for j in range(len(metadata['filepaths'])):
        filepath = metadata['filepaths'][j]
        csv_data = CSVParser.parse_csv(filepath, metadata['headers'][j], metadata['group_by'][j])
        if isinstance(csv_data, dict):
            for key, val in csv_data.items():
                if graph_type == GRAPH_TYPES.LINE.value:
                    lines.append(ax.plot(val[0], val[1], label=key, c=colors[j][key]))
                elif graph_type == GRAPH_TYPES.SCATTER.value:
                    lines.append(ax.scatter(val[0], val[1], label=key, c=colors[j][key]))
        else:
            if graph_type == GRAPH_TYPES.LINE.value:
                lines.append(ax.plot(csv_data[0], csv_data[1], label=metadata['names'][j], c=colors[filepath]))
            elif graph_type == GRAPH_TYPES.SCATTER.value:
                lines.append(ax.scatter(csv_data[0], csv_data[1], label=metadata['names'][j], c=colors[filepath]))
    return lines

def live_plot(metadata: Dict[Any, Any], title: str, labels: List[str], graph_type: str) -> None:
    print("Live Viewing Data (graph should appear as a new window)...")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(title)

    filepaths = metadata['filepaths']
    headers = metadata['headers']
    names = metadata['names']
    group_by = metadata['group_by']

    colors = color_dict(filepaths)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    for i in range(len(filepaths)):
        csv_data = CSVParser.parse_csv(filepaths[i], headers[i], group_by[i])
        if isinstance(csv_data, dict):
            colors[i] = color_dict(list(csv_data.keys()))
            for key, val in csv_data.items():
                if graph_type == GRAPH_TYPES.LINE.value:
                    ax.plot(val[0][0], val[1][0], c=colors[i][key], label=key)
                elif graph_type == GRAPH_TYPES.SCATTER.value:
                    ax.scatter(val[0][0], val[1][0], c=colors[i][key], label=key)
        elif isinstance(csv_data, list):        
            if graph_type == GRAPH_TYPES.LINE.value:
                ax.plot(csv_data[0][0], csv_data[1][0], c=colors[filepaths[i]], label=names[i])
            elif graph_type == GRAPH_TYPES.SCATTER.value:
                ax.scatter(csv_data[0][0], csv_data[1][0], c=colors[filepaths[i]], label=names[i])
    plt.legend()
    try:
        anim = animation.FuncAnimation(fig, update, fargs=(ax, metadata, colors, title, labels, graph_type), interval=100)
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(e)        
        sys.exit(0)
