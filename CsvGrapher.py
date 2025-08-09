import os
import argparse
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import csv
import sys
import glob
import numpy as np
import random
import matplotlib.animation as animation
from math import floor
from copy import copy, deepcopy
from enum import Enum
from typing import Dict, List, Any, Tuple

# Project files
from parser import DataParser, CSVParser
from metrics import PerformanceMetrics, ATE, METRIC_TYPES

# Log path configuration
currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')
graphs_path = os.path.join(log_path, 'graphs')
animated_path = os.path.join(graphs_path, 'animated')

if not os.path.exists(animated_path):
    os.makedirs(animated_path)

# Enumerations
class GRAPH_TYPES(Enum):
    LINE = "line"
    LINE3D = "line3d" 
    SCATTER = "scatter"
    SCATTER3D = "scatter3d"
    SCATTERH = "scatterh"
    HISTOGRAM = "hist"
    STEMPLOT = "stem"
    LINE_XX = "line_xx"
    LINE_YY = "line_yy"
    SCATTER_XX = "scatter_xx"
    SCATTER_YY = "scatter_yy"
    PERF2D = "perf2d"
    PERF3D = "perf3d"

# C++ - like Structures
class Parameters:
    __slots__ = ("title", "labels", "data", "graph_type", "save_file", "is_animated", "is_live", "metrics_config")
    def __init__(self) -> None:
        self.title = ''
        self.labels = list()
        self.data = dict()
        self.graph_type = GRAPH_TYPES.LINE.value
        self.save_file = False
        self.is_animated = False
        self.is_live = False
        self.metrics_config = dict()

# Number of axes used based on graph type 
graph_axes = {
    GRAPH_TYPES.LINE.value: 2,
    GRAPH_TYPES.LINE3D.value: 3, 
    GRAPH_TYPES.SCATTER.value: 2,
    GRAPH_TYPES.SCATTER3D.value: 3,
    GRAPH_TYPES.SCATTERH.value: 2,
    GRAPH_TYPES.HISTOGRAM.value: 2,
    GRAPH_TYPES.STEMPLOT.value: 2,
    GRAPH_TYPES.LINE_YY.value: 2,
    GRAPH_TYPES.SCATTER_YY.value: 2,
    GRAPH_TYPES.PERF2D.value: 2,
    GRAPH_TYPES.PERF3D.value: 3
}

###################
# General Methods #
###################

def plot(params: Parameters) -> None:
    if params.is_animated:
        print('Generating an animated gif may take some time (based on the quantity of data).')
    if params.graph_type in [GRAPH_TYPES.LINE.value,GRAPH_TYPES.LINE3D.value]:
        multi_line(params)
    elif params.graph_type in [GRAPH_TYPES.LINE_XX.value,GRAPH_TYPES.LINE_YY.value,GRAPH_TYPES.SCATTER_XX.value,GRAPH_TYPES.SCATTER_YY.value]:
        second_axis(params)
    elif params.graph_type in [GRAPH_TYPES.SCATTER.value,GRAPH_TYPES.SCATTER3D.value]:
        multi_scatter(params)
    elif params.graph_type == GRAPH_TYPES.SCATTERH.value:
        multi_scatter_histogram(params)
    elif params.graph_type == GRAPH_TYPES.HISTOGRAM.value:
        multi_histogram(params)
    elif params.graph_type == GRAPH_TYPES.STEMPLOT.value:
        multi_stem(params)
    elif params.graph_type in [GRAPH_TYPES.PERF2D.value, GRAPH_TYPES.PERF3D.value]:
        position_perf(params)
    else:
        print("Unrecognized graph type: %s"%(params.graph_type))
        sys.exit(0)

def color_dict(data: List[str]) -> Dict[Any, Any]:
    tmp = dict()
    for key in data:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hexcolor = '#%02x%02x%02x'%(r, g, b)
        tmp[key] = hexcolor
    return tmp

def get_limits(data: Dict[Any, Any]) -> List[Any]:
    tmp = []
    for val in list(data.values())[0]:
        tmp.append([float('inf'), -1*float('inf')])
    for val in data.values():
        for i in range(len(val)):
            tmp[i][0] = min(val[i]) if tmp[i][0] > min(val[i]) else tmp[i][0]
            tmp[i][1] = max(val[i]) if tmp[i][1] < max(val[i]) else tmp[i][1]
    for lim in tmp:
        diff = abs(lim[1] - lim[0])
        offset = 0.15*diff if diff > 0 else 0.5
        lim[0] = lim[0] - offset
        lim[1] = lim[1] + offset

    return tmp

def create_filename(title: str) -> str:
    import re
    from datetime import datetime
    stamp = datetime.now().isoformat('_', timespec='seconds')
    filename = '%s_%s'%(stamp, title)
    filename = re.sub('\-|\:|\s+', '_', filename)
    return filename

def save(title: str, anim:Any=None) -> None:
    filename = create_filename(title)
    if anim is not None:
        writergif = animation.PillowWriter(fps=60)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        filepath = os.path.join(graphs_path, filename)
        plt.savefig(filepath)
        print("%s.png has been created."%(filepath))

####################
# Animated Methods #
####################

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

def animate_graph(fig: Figure, ax: Axes, title: str, graph_type: str, data: Dict[str, Any]) -> None:
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
    save(title, anim)

#####################
# Live View Methods #
#####################

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

####################
# Graphing Methods #
####################

def second_axis(p: Parameters) -> None:
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
        save(p.title)
    else:
        plt.show()

def multi_line(p: Parameters) -> None:
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
        animate_graph(fig=fig,ax=ax,title=p.title,graph_type=p.graph_type,data=p.data)
    else:
        for key, val in p.data.items():
            if num_of_axes == 2:
                ax.plot(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if p.save_file:
            save(p.title)
        else:
            plt.show()

def multi_scatter(p: Parameters) -> None:
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
        animate_graph(fig=fig,ax=ax,title=p.title,graph_type=p.graph_type,data=p.data)
    else:
        for key, val in p.data.items():
            if num_of_axes == 2:
                ax.scatter(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.scatter(val[0], val[1], val[2], label=key)
        plt.legend()
        if p.save_file:
            save(p.title)
        else:
            plt.show()

def multi_scatter_histogram(p: Parameters) -> None:
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
        save(p.title)
    else:
        plt.show()

def multi_histogram(p: Parameters) -> None:
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
        save(p.title)
    else:
        plt.show()

def multi_stem(p: Parameters) -> None:
    fig, ax = plt.subplots()
    plt.title(p.title)

    ax.set(xlabel=p.labels[0], ylabel=p.labels[1])
    ax.autoscale(enable=True, axis='both')

    colors = color_dict(list(p.data.keys()))
    for key, val in p.data.items():
        ax.stem(val[0], val[1], label=key, linefmt=colors[key], markerfmt='D')
    plt.legend()

    if p.save_file:
        save(p.title)
    else:
        plt.show()

def position_perf(p: Parameters) -> None:
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
        save(p.title)
    else:
        plt.show()

###########
# Metrics #
###########

def show_ate_results(metric_name: str, expected: Any, path: Any, p: Parameters) -> None:
    data_pos = []
    data_x  = []
    data_y = []
    keys = list(p.data.keys())
    ground_truth = p.data[expected]
    row_labels = []
    for i in range(0,len(keys)):
        if keys[i] != expected:
            method = p.data[keys[i]]
            metrics = ATE.get_pose_diff_metrics(ground_truth, method, keys[i])
            data_pos.append(metrics)
            metrics = ATE.get_single_val_metrics(ground_truth, method, "x")
            data_x.append(metrics)
            metrics = ATE.get_single_val_metrics(ground_truth, method, "y")
            data_y.append(metrics)
            row_labels.append(keys[i])

    collection = [data_pos, data_x, data_y]
    #print(collection)
    order = ["Position", "X", "Y"]
    cols = ["RSME", "Mean", "Standard Deviation"]
    filename = create_filename("%s_ATE"%p.title)
    PerformanceMetrics.show_table(f"%s_ATE"%p.title, filename, row_labels, cols, collection, order, path, p.save_file)
    print("Complete")

def show_metric(metric_name: str, metric_type: str, ground_truth: Any, p: Parameters) -> None:
    if metric_type == METRIC_TYPES.ATE.value:
        show_ate_results(metric_name, ground_truth, graphs_path, p)

################
# Main Program #
################

def main():
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file (leave blank if parent directory is log/).')
    parser.add_argument('-f', '--file', action='store', type=str, help='Desired CSV file.')
    parser.add_argument('-c', '--column-headers', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-b', '--group-by', action='store', type=str, help="Group the data based on a specific column name.")
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line_xx", "line_yy", "line3d", "scatter", "scatter_xx", "scatter_yy", "scatter3d", "scatterh", "hist", "stem", "perf2d", "perf3d"]. Default: \"line\".', default=GRAPH_TYPES.LINE.value)
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for the generated graph.')
    parser.add_argument('-l', '--live-view', action='store_true', help='Stream data from CSV files to Graph in real-time.')
    parser.add_argument('-a', '--animated', action='store_true', help='Creates an animated graph when true (will be saved as a gif).')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file.')

    args = parser.parse_args()
    params = Parameters()

    if args.yaml is not None:
        import yaml 
        filepath = ''  
        try:
            if args.yaml[-5:] != '.yaml':
                filepath = os.path.join(currentFolder, "%s.yaml"%(args.yaml))
            else:
                filepath = os.path.join(currentFolder, args.yaml)
            with open(filepath, 'r') as f:
                yf = yaml.safe_load(f)
                params.is_live = yf['live'] if 'live' in yf.keys() else False
                params.graph_type = yf['type'] if 'type' in yf.keys() else GRAPH_TYPES.LINE.value
                if params.is_live:
                    params.data = {'filepaths': [], 'headers': [], 'names': [], 'group_by': []}
                for file in yf['files']:
                    key = list(file.keys())[0]
                    val = file[key]
                    assigned_axis = val['axis'] if 'axis' in val else None
                    if 'path' in val: 
                        path = os.path.join(log_path, val['path'])
                    else:
                        path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'])
                    filepath = DataParser.get_file(path, val['name'])
                    group_by = val['group_by'] if 'group_by' in val else ''
                    print("Fetched %s"%(filepath))
                    if params.is_live:
                        params.data['filepaths'].append(filepath)
                        params.data['headers'].append(val['headers'])
                        params.data['names'].append(key)
                        params.data['group_by'].append(group_by)
                    else:
                        csv_data = CSVParser.parse_csv(filepath, val['headers'], group_by)
                        if isinstance(csv_data, dict):
                            for group_name in csv_data.keys():
                                params.data[group_name] = csv_data[group_name]
                        elif isinstance(csv_data, list):
                            if params.graph_type in [GRAPH_TYPES.LINE_XX.value,GRAPH_TYPES.SCATTER_XX.value]:
                                params.data[key] = tuple([assigned_axis, csv_data]) # type: ignore
                            elif params.graph_type in [GRAPH_TYPES.LINE_YY.value,GRAPH_TYPES.SCATTER_YY.value]:  
                                params.data[key] = tuple([assigned_axis, csv_data]) # type: ignore
                            else:
                                params.data[key] = csv_data
                params.labels = [yf['labels']['x_label'], yf['labels']['y_label']]
                if 'z_label' in yf['labels']:
                    params.labels.append(yf['labels']['z_label'])
                params.title = yf['title'] if 'title' in yf.keys() else ''
                params.is_animated = yf['animated'] if 'animated' in yf.keys() else False
                params.save_file = yf['save'] if 'save' in yf.keys() else False

                if params.is_live:
                    live_plot(metadata=params.data,title=params.title,labels=params.labels,graph_type=params.graph_type)
                else:
                    plot(params)

                if 'metrics' in yf:
                    for metric in yf['metrics']:
                        key = list(metric.keys())[0]
                        definition = metric[key]
                        params.metrics_config[key] = definition    
                    for key, val in params.metrics_config.items():
                        show_metric(key, val['type'], val['ground_truth'], params)

        except FileNotFoundError:
            print('YAML file: %s does not exist.'%(filepath))
        except KeyError as k:
            print("KeyError: %s"%(k))
        except TypeError as t:
            print("One of the fields in your YAML file is not filled properly.\n%s"%(t))
    else:
        filename = args.file
        path_to_file = args.path if args.path is not None else ''
        filepath = DataParser.get_file(os.path.join(log_path, path_to_file), filename) 
        params.title = args.title if args.title is not None else filename[:-4]
        params.graph_type = args.graph_type if args.graph_type is not None else GRAPH_TYPES.LINE.value
        group_by = args.group_by if args.group_by != '' else ''
        params.labels = args.column_headers if args.column_headers is not None else []
        params.is_live = args.live_view if args.live_view is not None else False
        if params.is_live:
            params.data = {'filepaths':[filepath], 'headers':[params.labels], 'names':[filename[:-4]], 'group_by':[group_by]}
            live_plot(metadata=params.data,title=params.title,labels=params.labels,graph_type=params.graph_type)
        else:
            params.save_file = args.save if args.save is not None else False
            params.is_animated = args.animated if args.animated is not None else False
            values = CSVParser.parse_csv(filepath, args.column_headers, group_by)
            params.data = {filename: values} if group_by is not None else values # type: ignore
            plot(params) 

if __name__ == "__main__":
    main()