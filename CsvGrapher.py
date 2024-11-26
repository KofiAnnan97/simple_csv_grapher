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

from metrics import PerformanceMetrics

"""class DataParser:
    def __init__(self):
        self.labels = []
        self.data = dict()
        self.save_file = False
        self.is_animated = False
        self.live_view = False

    def get_file(self, path, filename):
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

    def get_col_idxs(self, col_names, headers):
        tmp = []
        for col in col_names:
            try:
                idx = headers.index(col)
                tmp.append(idx)
            except ValueError:
                raise ValueError('Column header \'%s\' is not in the csv file.'%(col))
        return tmp

    def parse_csv(filepath, col_names):
        vals = []
        try:
            with open(filepath, 'r') as cr:
                reader = csv.reader(cr)
                headers = next(reader)
                col_idxs = get_col_idxs(col_names, headers)

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
        if len(vals[0]) == 0:
            raise Exception('%s has no data.'%(filepath))
        return vals"""

currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')
graphs_path = os.path.join(log_path, 'graphs')
animated_path = os.path.join(graphs_path, 'animated')

if not os.path.exists(animated_path):
    os.makedirs(animated_path)

graph_axes = {'line': 2,
              'line3d': 3, 
              'scatter': 2,
              'scatter3d': 3,
              'scatterh': 2,
              'hist': 2,
              'stem': 2}

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
    try:
        with open(filepath, 'r') as cr:
            if group_by is not None:
                reader = csv.reader(cr)
                headers = next(reader)
                col_names.insert(0, group_by)
                col_idxs = get_col_idxs(col_names, headers)
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
                col_idxs = get_col_idxs(col_names, headers)
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
    if (isinstance(vals, list) and len(vals[0]) == 0) or (isinstance(vals, dict) and len(vals.keys()) == 0):
        raise Exception('%s has no data.'%(filepath))
    return vals

def plot(graph_type, title, labels, data, is_animated, save_file):
    if is_animated == True:
        print('Generating an animated gif may take some time (based on the quantity of data).')
    if graph_type in ['line', 'line3d']:
        multi_line(title,labels,data,graph_type,is_animated,save_file)
    elif graph_type == 'line_yy':
        line_yy(title, labels, data, save_file)
    elif graph_type in ['scatter', 'scatter3d']:
        multi_scatter(title,labels,data,graph_type,is_animated,save_file)
    elif graph_type =='scatterh':
        multi_scatter_histogram(title,labels,data,save_file)
    elif graph_type == "hist":
         multi_histogram(title,labels,data,save_file)
    elif graph_type == 'stem':
        multi_stem(title,labels,data,save_file)
    elif graph_type == "perf2d":
        perf2d(title,labels,data,save_file)
    elif graph_type == "perf3d":
        perf3d(title,labels,data,save_file)
    else:
        print("Unrecognized graph type: %s"%(graph_type))
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

def animate(i, ax, data, idx_size, graph_type, colors):
    try:
        lines = None
        if graph_type == 'line':
            lines = [ax.plot(val[0][:i*idx_size], val[1][:i*idx_size], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == 'scatter':
            lines = [ax.scatter(val[0][:i*idx_size], val[1][:i*idx_size], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == 'line3d':
            lines = [ax.plot(val[0][:i*idx_size], val[1][:i*idx_size], val[2][:i*idx_size], label=key, c=colors[key]) for key, val in data.items()]
        elif graph_type == 'scatter3d':
            lines = [ax.scatter(val[0][:i*idx_size], val[1][:i*idx_size], val[2][:i*idx_size], label=key, c=colors[key]) for key, val in data.items()]  
        return lines,
    except Exception as e:
        print(e)

def animate_graph(fig, ax, graph_name, g_type, data):
    colors = color_dict(list(data.keys()))
    for key, val in data.items():
        if g_type == 'line':
            ax.plot(val[0][0], val[1][0], c=colors[key], label=key)
        elif g_type == 'line3d':
            ax.plot(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        elif g_type == 'scatter':
            ax.scatter(val[0][0], val[1][0], c=colors[key], label=key)
        elif g_type == 'scatter3d':
            ax.scatter(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
    plt.legend()
    first = list(data.keys())[0]
    data_length = len(data[first][0])
    index_size = floor(0.0067 * data_length) if floor(0.0067 * data_length) >= 1 else 1 
    frame_size = data_length//index_size
    anim = animation.FuncAnimation(fig, animate, frames=frame_size,fargs=(ax, data, index_size, g_type, colors), interval=1, repeat=False)
    save(graph_name, anim)


##########################
# Live Processing Methods#
##########################

def update(i, ax, metadata, labels, title, colors, graph_type):
    plt.title(title)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    lines = []
    for j in range(len(metadata['filepaths'])):
        filepath = metadata['filepaths'][j]
        data = parse_csv(filepath, metadata['headers'][j], None)
        if graph_type == 'line':
            lines.append(ax.plot(data[0], data[1], label=metadata['names'], c=colors[filepath]))
        elif graph_type == 'scatter':
            lines.append(ax.scatter(data[0], data[1], label=metadata['names'], c=colors[filepath]))
    return lines

def live_plot(graph_type, graph_name, metadata, labels):
    print("Live Viewing Data (graph should appear as a new window)...")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(graph_name)

    filepaths = metadata['filepaths']
    headers = metadata['headers']
    names = metadata['names']

    colors = color_dict(filepaths)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    for i in range(len(filepaths)):
        data = parse_csv(filepaths[i], headers[i], None)
        if graph_type == 'line':
            ax.plot(data[0][0], data[1][0], c=colors[filepaths[i]], label=names[i])
        elif graph_type == 'scatter':
            ax.scatter(data[0][0], data[1][0], c=colors[filepaths[i]], label=names[i])
    plt.legend()
    try:
        anim = animation.FuncAnimation(fig, update, fargs=(ax, metadata, labels, graph_name, colors, graph_type), interval=100)
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(e)        
        sys.exit(0)

####################
# Graphing Methods #
####################

def line_yy(graph_name, labels, data, save_file):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(labels[0])
    colors = color_dict(['y_axis1', 'y_axis2'])
    ax1.set_ylabel(labels[1], color=colors['y_axis1'])
    ax2 = ax1.twinx()
    ax2.set_ylabel(labels[2], color=colors['y_axis2'])
    plt.title(graph_name)

    plots = []
    for key, val in data.items():
        points = val[1]
        #print(val[0])
        if val[0] == 1:
            plot = ax1.plot(points[0], points[1], label=key, color=colors['y_axis1'])
        elif val[0] == 2:
            plot = ax2.plot(points[0], points[1], label=key, color=colors['y_axis2'])
        plots += plot
    labels = [l.get_label() for l in plots]
    plt.legend(plots, labels,loc='upper right')
    if save_file == True:
        save(graph_name)
    else:
        plt.show()
    

def multi_line(graph_name, labels, data, g_type, is_animated, save_file):
    num_of_axes = graph_axes[g_type]
    fig = plt.figure()
    if num_of_axes == 2:
        ax = fig.add_subplot(1, 1, 1)
    elif num_of_axes == 3:
        ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animate_graph(fig=fig,ax=ax,graph_name=graph_name, g_type=g_type, data=data)
    else:
        for key, val in data.items():
            if num_of_axes == 2:
                ax.plot(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_scatter(graph_name, labels, data, g_type, is_animated, save_file):
    num_of_axes = graph_axes[g_type]
    fig = plt.figure() 
    if num_of_axes == 2:
        ax = fig.add_subplot(1, 1, 1)
    elif num_of_axes == 3:
        ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animate_graph(fig=fig,ax=ax,graph_name=graph_name, g_type=g_type, data=data)
    else:
        for key, val in data.items():
            if num_of_axes == 2:
                ax.scatter(val[0], val[1], label=key)
            elif num_of_axes == 3:
                ax.scatter(val[0], val[1], val[2], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_scatter_histogram(graph_name, labels, data, save_file):
    print("This figure may take a while to complete based on the size of data.")
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    fig.suptitle(graph_name)

    limits = get_limits(data)
    xmin, xmax = limits[0]
    ymin, ymax = limits[1]
    x_binwidth = (xmax - xmin)/20.0
    y_binwidth = (ymax - ymin)/20.0

    x_bins = np.arange(xmin, xmax + x_binwidth, x_binwidth)
    y_bins = np.arange(ymin, ymax + y_binwidth, y_binwidth)

    for key, val in data.items():
        ax.scatter(val[0], val[1], label=key)
        ax_histx.hist(val[0], bins=x_bins)
        ax_histy.hist(val[1], bins=y_bins, orientation='horizontal')

    plt.legend()
    if save_file == True:
        save(graph_name)
    else:
        plt.show()

def multi_histogram(graph_name, labels, data, save_file):
    n_bins = 20

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.suptitle(graph_name)
    for key, val in data.items():
        axs[0].hist(val[0], bins=n_bins, label=key)
        axs[1].hist(val[1], bins=n_bins, label=key)

    axs[0].set_xlabel(labels[0])
    axs[1].set_xlabel(labels[1])
    axs[0].autoscale(enable=True, axis='both')
    axs[1].autoscale(enable=True, axis='both') 
    plt.legend()  

    if save_file == True:
        save(graph_name)
    else:
        plt.show()

def multi_stem(graph_name, labels, data, save_file):
    fig, ax = plt.subplots()
    plt.title(graph_name)

    ax.set(xlabel=labels[0], ylabel=labels[1])
    ax.autoscale(enable=True, axis='both')

    colors = color_dict(list(data.keys()))
    for key, val in data.items():
        ax.stem(val[0], val[1], label=key, linefmt=colors[key], markerfmt='D')
    plt.legend()

    if save_file == True:
        save(graph_name)
    else:
        plt.show()

def show_ate_results(graph_name, data, ref_key, path, save_file):
    pm = PerformanceMetrics()
    data_pos = []
    data_x  = []
    data_y = []
    keys = list(data.keys())
    ground_truth = data[ref_key]
    row_labels = []
    for i in range(0,len(keys)):
        if keys[i] != ref_key:
            method = data[keys[i]]
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
    filename = create_filename("%s_ATE"%graph_name)
    pm.show_table(f"{graph_name} ATE", filename, row_labels, cols, collection, order, path, save_file)
    print("Complete")

def perf2d(graph_name, labels, data, save_file):
    ref_key = "GPS"
    fig, axs = plt.subplots(2, sharex=True)
    plt.suptitle(graph_name)
    ylabels = labels[1].split(",")

    limits = get_limits(data)
    axs[0].set(xlim=limits[0], ylim=limits[1], ylabel=ylabels[0])
    axs[1].set(xlim=limits[0], ylim=limits[2], xlabel=labels[0], ylabel=ylabels[1])
   
    for key, val in data.items():
        axs[0].plot(val[0], val[1], label=key)
        axs[1].plot(val[0], val[2], label=key)
    
    plt.legend()
    if save_file == True:
        save(graph_name)
    else:
        plt.show()
    show_ate_results(graph_name, data, ref_key, graphs_path, save_file)

def perf3d(graph_name, labels, data, save_file):
    fig, axs = plt.subplots(3, sharex=True)
    plt.suptitle(graph_name)
    ylabels = labels[1].split(",")

    limits = get_limits(data)
    axs[0].set(xlim=limits[0], ylim=limits[1], ylabel=ylabels[0])
    axs[1].set(xlim=limits[0], ylim=limits[2], ylabel=ylabels[1])
    axs[2].set(xlim=limits[0], ylim=limits[3], xlabel=labels[0], ylabel=ylabels[2])
   
    for key, val in data.items():
        axs[0].plot(val[0], val[1], label=key)
        axs[1].plot(val[0], val[2], label=key)
        axs[2].plot(val[0], val[3], label=key)
    
    plt.legend()
    if save_file == True:
        save(graph_name)
    else:
        plt.show()

################
# Main Program #
################

def main():
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file (leave blank if parent directory is log/).')
    parser.add_argument('-f', '--file', action='store', type=str, help='Desired CSV file.')
    parser.add_argument('-c', '--column-headers', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-b', '--group-by', action='store', type=str, help="Group the data based on a specific column name.")
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line_yy", "line3d", "scatter", "scatter3d", "scatterh", "hist", "stem"]. Default: \'line\'.', default="line")
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for the generated graph.')
    parser.add_argument('-l', '--live-view', action='store_true', help='Stream data from CSV files to Graph in real-time.')
    parser.add_argument('-a', '--animated', action='store_true', help='Creates an animated graph when true (will be saved as a gif).')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file.')

    args = parser.parse_args()

    if args.yaml is not None:
        import yaml
        data = dict()
        try:
            if args.yaml[-5:] != '.yaml':
                filepath = os.path.join(currentFolder, "%s.yaml"%(args.yaml))
            else:
                filepath = os.path.join(currentFolder, args.yaml)
            with open(filepath, 'r') as f:
                yf = yaml.safe_load(f)
                is_live = yf['live'] if 'live' in yf.keys() else False
                type = yf['type'] if 'type' in yf.keys() else 'line'
                assigned_y_axis = None
                if is_live == True:
                    data = {'filepaths': [], 'headers': [], 'names': []}
                for file in yf['files']:
                    key = list(file.keys())[0]
                    val = file[key]
                    if 'path' in val: 
                        path = os.path.join(log_path, val['path'])
                    else:
                        path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'])
                    filepath = get_file(path, val['name'])
                    group_by = val['group_by'] if 'group_by' in val else None
                    csv_data = parse_csv(filepath, val['headers'], group_by)
                    print("Fetched %s"%(filepath))
                    if 'y_axis' in val:
                        assigned_y_axis = val['y_axis']
                    if is_live == True:
                        data['filepaths'].append(filepath)
                        data['headers'].append(val['headers'])
                        data['names'].append(key)
                    else:
                        if isinstance(csv_data, dict):
                            for group_name in csv_data.keys():
                                data[group_name] = csv_data[group_name]
                        elif isinstance(csv_data, list):
                            if type == 'line_yy':            
                                data[key] = tuple([assigned_y_axis, csv_data])
                            else:
                                data[key] = csv_data
                labels = [yf['labels']['x_label'], yf['labels']['y_label']]
                if 'y2_label' in yf['labels']:
                    labels.append(yf['labels']['y2_label']) 
                if 'z_label' in yf['labels']:
                    labels.append(yf['labels']['z_label'])
                title = yf['title'] if 'title' in yf.keys() else ''
                animated = yf['animated'] if 'animated' in yf.keys() else False
                save_graph = yf['save'] if 'save' in yf.keys() else False
                if is_live == True:
                    live_plot(graph_type=type,graph_name=title, metadata=data, labels=labels)
                else:
                    plot(graph_type=type, title=title, labels=labels, data=data, is_animated=animated, save_file=save_graph)
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
        title = args.title if args.title is not None else filename[:-4]
        graph_type = args.graph_type if args.graph_type is not None else 'line'
        group_by = args.group_by if args.group_by is not None else None
        if args.live_view == True:
            data = {'filepaths':[filepath], 'headers':[args.column_headers], 'names':[filename[:-4]]}
            live_plot(graph_type=graph_type,graph_name=title, labels=args.column_headers, metadata=data)
        else:
            values = parse_csv(filepath, args.column_headers, group_by)
            data = {filename: values} if group_by is None else values
            plot(graph_type=graph_type, title=title, labels=args.column_headers, data=data, is_animated=args.animated, save_file=args.save) 

if __name__ == "__main__":
    main()