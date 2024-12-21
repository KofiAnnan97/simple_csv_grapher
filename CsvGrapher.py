import os
import argparse
import matplotlib.pyplot as plt
import csv
import sys
import glob
import numpy as np
import random
import matplotlib.animation as animation
from math import floor
from copy import copy, deepcopy

from metrics import PerformanceMetrics

# Log path configuration
currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')
graphs_path = os.path.join(log_path, 'graphs')
animated_path = os.path.join(graphs_path, 'animated')

if not os.path.exists(animated_path):
    os.makedirs(animated_path)

# Data & Visualization Parameters
TITLE = ''
LABELS = list()
DATA = dict()
GRAPH_TYPE = 'line'
IS_ANIMATED = False
SAVE_FILE = False
IS_LIVE = False
METRICS_CONFIG = dict()

# Number of axes used based on graph type 
graph_axes = {'line': 2,
              'line3d': 3, 
              'scatter': 2,
              'scatter3d': 3,
              'scatterh': 2,
              'hist': 2,
              'stem': 2,
              'line_yy': 2,
              'scatter_yy': 2,
              'perf2d': 2,
              'perf3d': 3
            }

###################
# General Methods #
###################

def get_file(path, filename):
    if filename == 'latest':
        list_of_files = glob.glob(os.path.join(path, '*'), recursive=False)
        list_of_files.sort()
        filepath = max(list_of_files)
    elif filename == 'lastModified':
        list_of_files = glob.glob(os.path.join(path, '*'), recursive=False)
        filepath = max(list_of_files, key=os.path.getmtime)
    else:
        filepath = os.path.join(path, filename)
    return filepath

def get_col_idxs(col_names, headers):
    tmp = []
    for col in col_names:
        try:
            idx = headers.index(col)
            tmp.append(idx)
        except ValueError:
            raise ValueError('Column header \'%s\' is not in the csv file.'%(col))
    return tmp
    
def parse_csv(filepath, col_names, group_by=None):
    vals = []
    groups = dict()
    tmp_cols = copy(col_names)
    try:
        with open(filepath, 'r') as cr:
            if group_by is not None:
                reader = csv.reader(cr)
                headers = next(reader)
                tmp_cols.insert(0, group_by)
                col_idxs = get_col_idxs(tmp_cols, headers)
                for row in reader:
                    group_name = row[col_idxs[0]]
                    if group_name not in groups.keys():
                        tmp = []
                        for i in range(len(col_idxs)-1):
                            tmp.append(list())
                        groups[group_name] = tmp
                    for i in range(1, len(col_idxs)):
                        idx = col_idxs[i]
                        groups[group_name][i-1].append(float(row[idx]))
                vals = groups
            else:
                reader = csv.reader(cr)
                headers = next(reader)
                col_idxs = get_col_idxs(tmp_cols, headers)
                for i in range(len(col_idxs)):
                    vals.append(list())
                for row in reader:
                    for i in range(len(col_idxs)):
                        idx = col_idxs[i]
                        vals[i].append(float(row[idx]))
    except FileNotFoundError:
        raise FileNotFoundError('%s does not exist. Check that path and filename.'%(filepath))
    except StopIteration:
        raise StopIteration('%s is empty.'%(filepath))
    if (isinstance(vals, list) and len(vals) == 0) or (isinstance(vals, dict) and len(vals.keys()) == 0):
        raise Exception('%s has no data.'%(filepath))
    return vals

def plot():
    if IS_ANIMATED == True:
        print('Generating an animated gif may take some time (based on the quantity of data).')
    if GRAPH_TYPE in ['line', 'line3d']:
        multi_line()
    elif GRAPH_TYPE in ['line_xx', 'line_yy', 'scatter_xx', 'scatter_yy']:
        second_axis()
    elif GRAPH_TYPE in ['scatter', 'scatter3d']:
        multi_scatter()
    elif GRAPH_TYPE =='scatterh':
        multi_scatter_histogram()
    elif GRAPH_TYPE == "hist":
        multi_histogram()
    elif GRAPH_TYPE == 'stem':
        multi_stem()
    elif GRAPH_TYPE in ["perf2d", "perf3d"]:
        position_perf()
    else:
        print("Unrecognized graph type: %s"%(GRAPH_TYPE))
        sys.exit(0)

def color_dict(data):
    tmp = dict()
    for key in data:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hexcolor = '#%02x%02x%02x'%(r, g, b)
        tmp[key] = hexcolor
    return tmp

def get_limits(data):
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

def create_filename(title):
    import re
    from datetime import datetime
    stamp = datetime.now().isoformat('_', timespec='seconds')
    filename = '%s_%s'%(stamp, title)
    filename = re.sub('\-|\:|\s+', '_', filename)
    return filename

def save(title, anim=None):
    filename = create_filename(title)
    if anim is not  None:
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

def idx_sizes_dict(data):
    tmp = dict()
    largest_size = 0
    for key, values in data.items():
        data_length = len(values[0])
        largest_size = data_length if data_length > largest_size else largest_size
        index_size = floor(0.067 * data_length) if floor(0.067 * data_length) >= 1 else 1 
        tmp[key] = index_size
    return largest_size, tmp

def animate(i, ax, idx_sizes, colors):
    try:
        lines = None
        if GRAPH_TYPE == 'line':
            lines = [ax.plot(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in DATA.items()]
        elif GRAPH_TYPE == 'scatter':
            lines = [ax.scatter(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in DATA.items()]
        elif GRAPH_TYPE == 'line3d':
            lines = [ax.plot(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], val[2][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in DATA.items()]
        elif GRAPH_TYPE == 'scatter3d':
            lines = [ax.scatter(val[0][:i*idx_sizes[key]], val[1][:i*idx_sizes[key]], val[2][:i*idx_sizes[key]], label=key, c=colors[key]) for key, val in DATA.items()]  
        return lines,
    except Exception as e:
        print(e)

def animate_graph(fig, ax):
    colors = color_dict(list(DATA.keys()))
    for key, val in DATA.items():
        if GRAPH_TYPE == 'line':
            ax.plot(val[0][0], val[1][0], c=colors[key], label=key)
        elif GRAPH_TYPE == 'line3d':
            ax.plot(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        elif GRAPH_TYPE == 'scatter':
            ax.scatter(val[0][0], val[1][0], c=colors[key], label=key)
        elif GRAPH_TYPE == 'scatter3d':
            ax.scatter(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
    plt.legend()
    
    data_length, idx_sizes = idx_sizes_dict(DATA)
    index_size = floor(0.05 * data_length) if floor(0.05 * data_length) >= 1 else 1 
    frame_size = data_length//index_size

    anim = animation.FuncAnimation(fig, animate, frames=frame_size,fargs=(ax, idx_sizes, colors), interval=1, repeat=False)
    save(TITLE, anim)

#####################
# Live View Methods #
#####################

def update(i, ax, metadata, colors):
    plt.title(TITLE)
    ax.set(xlabel=LABELS[0], ylabel=LABELS[1])
    lines = []
    for j in range(len(metadata['filepaths'])):
        filepath = metadata['filepaths'][j]
        csv_data = parse_csv(filepath, metadata['headers'][j], metadata['group_by'][j])
        if isinstance(csv_data, dict):
            for key, val in csv_data.items():
                if GRAPH_TYPE == 'line':
                    lines.append(ax.plot(val[0], val[1], label=key, c=colors[j][key]))
                elif GRAPH_TYPE == 'scatter':
                    lines.append(ax.scatter(val[0], val[1], label=key, c=colors[j][key]))
        else:
            if GRAPH_TYPE == 'line':
                lines.append(ax.plot(csv_data[0], csv_data[1], label=metadata['names'][j], c=colors[filepath]))
            elif GRAPH_TYPE == 'scatter':
                lines.append(ax.scatter(csv_data[0], csv_data[1], label=metadata['names'][j], c=colors[filepath]))
    return lines

def live_plot(metadata):
    print("Live Viewing Data (graph should appear as a new window)...")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(TITLE)

    filepaths = metadata['filepaths']
    headers = metadata['headers']
    names = metadata['names']
    group_by = metadata['group_by']

    colors = color_dict(filepaths)
    ax.set(xlabel=LABELS[0], ylabel=LABELS[1])
    for i in range(len(filepaths)):
        csv_data = parse_csv(filepaths[i], headers[i], group_by[i])
        if isinstance(csv_data, dict):
            colors[i] = color_dict(list(csv_data.keys()))
            for key, val in csv_data.items():
                if GRAPH_TYPE == 'line':
                    ax.plot(val[0][0], val[1][0], c=colors[i][key], label=key)
                elif GRAPH_TYPE == 'scatter':
                    ax.scatter(val[0][0], val[1][0], c=colors[i][key], label=key)
        elif isinstance(csv_data, list):        
            if GRAPH_TYPE == 'line':
                ax.plot(csv_data[0][0], csv_data[1][0], c=colors[filepaths[i]], label=names[i])
            elif GRAPH_TYPE == 'scatter':
                ax.scatter(csv_data[0][0], csv_data[1][0], c=colors[filepaths[i]], label=names[i])
    plt.legend()
    try:
        anim = animation.FuncAnimation(fig, update, fargs=(ax, metadata, colors), interval=100)
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(e)        
        sys.exit(0)

####################
# Graphing Methods #
####################

def second_axis():
    fig, ax1 = plt.subplots()
    static_axis = None
    two_axes = None
    if GRAPH_TYPE in ['line_xx', 'scatter_xx']:
        static_axis = LABELS[1]
        two_axes = LABELS[0].split(",")[:2]
        ax1.set_ylabel(static_axis)
        colors = color_dict(two_axes)
        ax1.set_xlabel(two_axes[0], color=colors[two_axes[0]])
        ax2 = ax1.twiny()
        ax2.set_xlabel(two_axes[1], color=colors[two_axes[1]])
    elif GRAPH_TYPE in ['line_yy', 'scatter_yy']:
        static_axis = LABELS[0]
        two_axes = LABELS[1].split(",")[:2]
        ax1.set_xlabel(static_axis)
        colors = color_dict(two_axes)
        ax1.set_ylabel(two_axes[0], color=colors[two_axes[0]])
        ax2 = ax1.twinx()
        ax2.set_ylabel(two_axes[1], color=colors[two_axes[1]])
    plt.title(TITLE)

    plots = []
    for key, val in DATA.items():
        points = val[1]
        plot = None
        if val[0] == 1:
            if GRAPH_TYPE in ['line_xx', 'line_yy']:
                plot = ax1.plot(points[0], points[1], label=key, color=colors[two_axes[0]])
            elif GRAPH_TYPE in ['scatter_xx', 'scatter_yy']:
                plot = ax1.scatter(points[0], points[1], label=key, color=colors[two_axes[0]])
        elif val[0] == 2:
            if GRAPH_TYPE in ['line_xx', 'line_yy']:
                plot = ax2.plot(points[0], points[1], label=key, color=colors[two_axes[1]])
            elif GRAPH_TYPE in ['scatter_xx', 'scatter_yy']:
                plot = ax2.scatter(points[0], points[1], label=key, color=colors[two_axes[1]])
        if plot is not None:
            if GRAPH_TYPE in ['line_xx', 'line_yy']:
                plots += plot
            elif GRAPH_TYPE in ['scatter_xx', 'scatter_yy']:
                plots.append(plot)
    set_labels = [l.get_label() for l in plots]
    plt.legend(plots, set_labels,loc='upper right')
    if SAVE_FILE == True:
        save(TITLE)
    else:
        plt.show()

def multi_line():
    num_of_axes = graph_axes[GRAPH_TYPE]
    fig = plt.figure()
    if num_of_axes == 2:
        ax = fig.add_subplot(1, 1, 1)
    elif num_of_axes == 3:
        ax = plt.axes(projection='3d')
    plt.title(TITLE)

    limits = get_limits(DATA)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=LABELS[0], ylabel=LABELS[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=LABELS[0], ylabel=LABELS[1], zlabel=LABELS[2])

    if IS_ANIMATED == True:
        animate_graph(fig=fig,ax=ax)
    else:
        for key, val in DATA.items():
            if num_of_axes == 2:
                ax.plot(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if SAVE_FILE == True:
            save(TITLE)
        else:
            plt.show()

def multi_scatter():
    num_of_axes = graph_axes[GRAPH_TYPE]
    fig = plt.figure() 
    if num_of_axes == 2:
        ax = fig.add_subplot(1, 1, 1)
    elif num_of_axes == 3:
        ax = plt.axes(projection='3d')
    plt.title(TITLE)

    limits = get_limits(DATA)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=LABELS[0], ylabel=LABELS[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=LABELS[0], ylabel=LABELS[1], zlabel=LABELS[2])

    if IS_ANIMATED == True:
        animate_graph(fig=fig,ax=ax)
    else:
        for key, val in DATA.items():
            if num_of_axes == 2:
                ax.scatter(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.scatter(val[0], val[1], val[2], label=key)
        plt.legend()
        if SAVE_FILE == True:
            save(TITLE)
        else:
            plt.show()

def multi_scatter_histogram():
    print("This figure may take a while to complete based on the size of data.")
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax.set(xlabel=LABELS[0], ylabel=LABELS[1])
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    fig.suptitle(TITLE)

    limits = get_limits(DATA)
    xmin, xmax = limits[0]
    ymin, ymax = limits[1]
    x_binwidth = (xmax - xmin)/20.0
    y_binwidth = (ymax - ymin)/20.0

    x_bins = np.arange(xmin, xmax + x_binwidth, x_binwidth)
    y_bins = np.arange(ymin, ymax + y_binwidth, y_binwidth)

    for key, val in DATA.items():
        ax.scatter(val[0], val[1], label=key)
        ax_histx.hist(val[0], bins=x_bins)
        ax_histy.hist(val[1], bins=y_bins, orientation='horizontal')

    plt.legend()
    if SAVE_FILE == True:
        save(TITLE)
    else:
        plt.show()

def multi_histogram():
    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.suptitle(TITLE)
    for key, val in DATA.items():
        axs[0].hist(val[0], bins=n_bins, label=key)
        axs[1].hist(val[1], bins=n_bins, label=key)

    axs[0].set_xlabel(LABELS[0])
    axs[1].set_xlabel(LABELS[1])
    axs[0].autoscale(enable=True, axis='both')
    axs[1].autoscale(enable=True, axis='both') 
    plt.legend()  

    if SAVE_FILE == True:
        save(TITLE)
    else:
        plt.show()

def multi_stem():
    fig, ax = plt.subplots()
    plt.title(TITLE)

    ax.set(xlabel=LABELS[0], ylabel=LABELS[1])
    ax.autoscale(enable=True, axis='both')

    colors = color_dict(list(DATA.keys()))
    for key, val in DATA.items():
        ax.stem(val[0], val[1], label=key, linefmt=colors[key], markerfmt='D')
    plt.legend()

    if SAVE_FILE == True:
        save(TITLE)
    else:
        plt.show()

def position_perf():
    num_of_plots = graph_axes[GRAPH_TYPE]
    fig, axs = plt.subplots(num_of_plots, sharex=True)
    plt.suptitle(TITLE)
    ylabels = LABELS[1].split(",")

    limits = get_limits(DATA)
    axs[0].set(xlim=limits[0], ylim=limits[1], ylabel=ylabels[0])
    if num_of_plots == 2:
        axs[1].set(xlim=limits[0], ylim=limits[2], xlabel=LABELS[0], ylabel=ylabels[1])
    elif num_of_plots == 3:
        axs[1].set(xlim=limits[0], ylim=limits[2], ylabel=ylabels[1])
        axs[2].set(xlim=limits[0], ylim=limits[3], xlabel=LABELS[0], ylabel=ylabels[2])
    
    for key, val in DATA.items():
        axs[0].plot(val[0], val[1], label=key)
        axs[1].plot(val[0], val[2], label=key)
        if num_of_plots == 3:
            axs[2].plot(val[0], val[3], label=key)
    
    plt.legend()
    if SAVE_FILE == True:
        save(TITLE)
    else:
        plt.show()

###########
# Metrics #
###########

def show_metric(metric_name, metric_type, ground_truth):
    if metric_type == 'ate':
        show_ate_results(metric_name, ground_truth, graphs_path)

def show_ate_results(metric_name, expected, path):
    pm = PerformanceMetrics()
    data_pos = []
    data_x  = []
    data_y = []
    keys = list(DATA.keys())
    ground_truth = DATA[expected]
    row_labels = []
    for i in range(0,len(keys)):
        if keys[i] != expected:
            method = DATA[keys[i]]
            metrics = pm.get_pose_diff_metrics(ground_truth, method, keys[i])
            data_pos.append(metrics)
            metrics = pm.get_single_val_metrics(ground_truth, method, "x")
            data_x.append(metrics)
            metrics = pm.get_single_val_metrics(ground_truth, method, "y")
            data_y.append(metrics)
            row_labels.append(keys[i])

    collection = [data_pos, data_x, data_y]
    #print(collection)
    order = ["Position", "X", "Y"]
    cols = ('RSME', 'Mean', 'Standard Deviation')
    filename = create_filename("%s_ATE"%TITLE)
    pm.show_table(f"%s_ATE"%TITLE, filename, row_labels, cols, collection, order, path, SAVE_FILE)
    print("Complete")

################
# Main Program #
################

def main():
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file (leave blank if parent directory is log/).')
    parser.add_argument('-f', '--file', action='store', type=str, help='Desired CSV file.')
    parser.add_argument('-c', '--column-headers', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-b', '--group-by', action='store', type=str, help="Group the data based on a specific column name.")
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line_xx", "line_yy", "line3d", "scatter", "scatter_xx", "scatter_yy", "scatter3d", "scatterh", "hist", "stem", "perf2d", "perf3d"]. Default: \"line\".', default="line")
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for the generated graph.')
    parser.add_argument('-l', '--live-view', action='store_true', help='Stream data from CSV files to Graph in real-time.')
    parser.add_argument('-a', '--animated', action='store_true', help='Creates an animated graph when true (will be saved as a gif).')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file.')

    args = parser.parse_args()

    global TITLE, LABELS, DATA, GRAPH_TYPE, IS_ANIMATED, SAVE_FILE, IS_LIVE, METRICS_CONFIG
    
    if args.yaml is not None:
        import yaml   
        try:
            if args.yaml[-5:] != '.yaml':
                filepath = os.path.join(currentFolder, "%s.yaml"%(args.yaml))
            else:
                filepath = os.path.join(currentFolder, args.yaml)
            with open(filepath, 'r') as f:
                yf = yaml.safe_load(f)
                IS_LIVE= yf['live'] if 'live' in yf.keys() else False
                GRAPH_TYPE = yf['type'] if 'type' in yf.keys() else 'line'
                if IS_LIVE == True:
                    DATA = {'filepaths': [], 'headers': [], 'names': [], 'group_by': []}
                for file in yf['files']:
                    key = list(file.keys())[0]
                    val = file[key]
                    assigned_axis = val['axis'] if 'axis' in val else None
                    if 'path' in val: 
                        path = os.path.join(log_path, val['path'])
                    else:
                        path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'])
                    filepath = get_file(path, val['name'])
                    group_by = val['group_by'] if 'group_by' in val else None
                    print("Fetched %s"%(filepath))
                    if IS_LIVE == True:
                        DATA['filepaths'].append(filepath)
                        DATA['headers'].append(val['headers'])
                        DATA['names'].append(key)
                        DATA['group_by'].append(group_by)
                    else:
                        csv_data = parse_csv(filepath, val['headers'], group_by)
                        if isinstance(csv_data, dict):
                            for group_name in csv_data.keys():
                                DATA[group_name] = csv_data[group_name]
                        elif isinstance(csv_data, list):
                            if GRAPH_TYPE in ['line_xx', 'scatter_xx']:
                                DATA[key] = tuple([assigned_axis, csv_data])
                            elif GRAPH_TYPE in ['line_yy', 'scatter_yy']:  
                                DATA[key] = tuple([assigned_axis, csv_data])
                            else:
                                DATA[key] = csv_data
                LABELS = [yf['labels']['x_label'], yf['labels']['y_label']]
                if 'z_label' in yf['labels']:
                    LABELS.append(yf['labels']['z_label'])
                TITLE = yf['title'] if 'title' in yf.keys() else ''
                IS_ANIMATED = yf['animated'] if 'animated' in yf.keys() else False
                SAVE_FILE = yf['save'] if 'save' in yf.keys() else False

                if IS_LIVE == True:
                    live_plot(metadata=DATA)
                else:
                    plot()

                if 'metrics' in yf:
                    for metric in yf['metrics']:
                        key = list(metric.keys())[0]
                        definition = metric[key]
                        METRICS_CONFIG[key] = definition    
                    for key, val in METRICS_CONFIG.items():
                        show_metric(key, val['type'], val['ground_truth'])

        except FileNotFoundError:
            print('YAML file: %s does not exist.'%(filepath))
        except KeyError as k:
            print("KeyError: %s"%(k))
        except TypeError as t:
            print("One of the fields in your YAML file is not filled properly.\n%s"%(t))
    else:
        filename = args.file
        path_to_file = args.path if args.path is not None else ''
        filepath = get_file(os.path.join(log_path, path_to_file), filename) 
        TITLE = args.title if args.title is not None else filename[:-4]
        GRAPH_TYPE = args.graph_type if args.graph_type is not None else 'line'
        group_by = args.group_by if args.group_by is not None else None
        LABELS = args.column_headers if args.column_headers is not None else []
        IS_LIVE = args.live_view if args.live_view is not None else False
        if IS_LIVE == True:
            DATA = {'filepaths':[filepath], 'headers':[LABELS], 'names':[filename[:-4]], 'group_by':[group_by]}
            live_plot(metadata=DATA)
        else:
            SAVE_FILE = args.save if args.save is not None else False
            IS_ANIMATED = args.animated if args.animated is not None else False
            values = parse_csv(filepath, args.column_headers, group_by)
            DATA = {filename: values} if group_by is None else values
            plot() 

if __name__ == "__main__":
    main()