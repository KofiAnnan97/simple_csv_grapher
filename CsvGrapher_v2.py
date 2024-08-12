import os
import argparse
import matplotlib.pyplot as plt
import csv
import sys
import glob
import numpy as np
import random
import matplotlib.animation as animation

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

def get_col_idxs(col_names, header):
    tmp = []
    for col in col_names:
        idx = header.index(col)
        tmp.append(idx)
    return tmp

def parse_csv(filepath, col_names):
    vals = []
    try:
        with open(filepath, 'r') as cr:
            reader = csv.reader(cr)
            header = next(reader)
            col_idxs = get_col_idxs(col_names, header)

            for i in range(len(col_idxs)):
                vals.append(list())
            for row in reader:
                for i in range(len(col_idxs)):
                    idx = col_idxs[i]
                    vals[i].append(float(row[idx]))
    except FileNotFoundError:
        print('%s does not exist. Check that path and filename.'%(filepath))
        sys.exit(0)
    
    return vals

def plot(graph_type, title, labels, data, is_animated, save_file):
    num_of_axes = graph_axes[graph_type]
    if is_animated == True:
        print('Generating an animated gif may take a while based on the amount of data.')
    if graph_type == 'line':
        multi_line(title,labels,data,num_of_axes,is_animated,save_file)
    elif graph_type == 'line3d':
        multi_line(title,labels,data,num_of_axes,is_animated,save_file)
    elif graph_type == 'scatter':
        multi_scatter(title,labels,data,num_of_axes,is_animated,save_file)
    elif graph_type == 'scatter3d':
        multi_scatter(title,labels,data,num_of_axes,is_animated,save_file)
    elif graph_type =='scatterh':
        multi_scatter_histogram(title,labels,data,save_file)
    elif graph_type == "hist":
         multi_histogram(title,labels,data,save_file)
    elif graph_type == 'stem':
        multi_stem(title,labels,data,save_file)
    else:
        print("Unrecognized graph type: %s"%(graph_type))
        sys.exit(0)

def create_filename(title):
    import re
    from datetime import datetime
    stamp = datetime.now().isoformat('_', timespec='seconds')
    filename = '%s_%s'%(stamp, title)
    filename = re.sub('\-|\:|\s+', '_', filename)
    return filename

def save(title):
    filename = create_filename(title)
    filepath = os.path.join(graphs_path, filename)
    plt.savefig(filepath)
    print("%s.png has been created."%(filepath))

####################
# Animated Methods #
####################

def init(data, num_of_axes):
    tmp = dict()
    for key,val in data.items():
        tmp[key] = list()
        for i in range(num_of_axes):
           tmp[key].append(list())
    return tmp

def color_init(data):
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
        offset = 0.15*abs(lim[1] - lim[0])
        lim[0] = lim[0] - offset
        lim[1] = lim[1] + offset
    return tmp

def update_lines(i, ax, data, animated_data, colors, num_of_axes):
    try:
        for key, val in data.items():
            for j in range(num_of_axes):
                animated_data[key][j].append(val[j][i])  
            if num_of_axes == 2:
                lines = [ax.plot(x[:i], y[:i], label=key, c=colors[key]) for x, y in animated_data.values()]
            elif num_of_axes == 3:
                lines = [ax.plot(x[:i], y[:i], z[:i], label=key, c=colors[key]) for x, y, z in animated_data.values()]
        return lines
    except Exception as e:
        print(e)

def update_scatter(i, ax, data, animated_data, colors, num_of_axes):
    try:
        for key, val in data.items():
            for j in range(num_of_axes):
                animated_data[key][j].append(val[j][i])
            if num_of_axes == 2:  
                lines = [ax.scatter(x[:i], y[:i], label=key, c=colors[key]) for x, y in animated_data.values()]
            elif num_of_axes == 3:
                lines = [ax.scatter(x[:i], y[:i], z[:i], label=key, c=colors[key]) for x, y,z in animated_data.values()]        
        return lines
    except Exception as e:
        print(e)

##########################
# Live Processing Methods#
##########################

def animate(i, ax, filepaths, headers, labels, name, colors):
    plt.title(name)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    for j in range(len(filepaths)):
        filepath = filepaths[j]
        data = parse_csv(filepath, headers[j])
        lines = [ax.plot(data[0], data[1], label=filepath, c=colors[filepath])]
    #plt.pause(0.1)
    return lines

def live_plot(graph_name, labels, filepaths, headers):
    print("Live Viewing Data (graph should appear as a new window)...")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(graph_name)
    colors = color_init(filepaths)
    ax.set(xlabel=labels[0], ylabel=labels[1])
    try:
        anim = animation.FuncAnimation(fig, animate, fargs=(ax, filepaths, headers, labels, graph_name, colors), interval=100)
        plt.show()
    except Exception as e:
        print(e)
        sys.exit(0)

####################
# Graphing Methods #
####################

def multi_line(graph_name, labels, data, num_of_axes, is_animated, save_file):
    if num_of_axes == 2:
        fig, ax = plt.subplots()
    elif num_of_axes == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animated_data = init(data, num_of_axes)
        colors = color_init(list(data.keys()))
        if num_of_axes == 2:
            for key, val in data.items():
                ax.plot(val[0][0], val[1][0], c=colors[key], label=key)
        elif num_of_axes == 3:
            for key, val in data.items():
                ax.plot(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_lines, frames=len(data[first][0]),fargs=(ax, data, animated_data, colors, num_of_axes), interval=100)
        writergif = animation.PillowWriter(fps=60)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        if num_of_axes == 2:
            for key, val in data.items():
                ax.plot(val[0], val[1], label=key)
        elif num_of_axes == 3:
            for key, val in data.items():
                ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_scatter(graph_name, labels, data, num_of_axes, is_animated, save_file):
    if num_of_axes == 2:
        fig, ax = plt.subplots()
    elif num_of_axes == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    if num_of_axes == 2:
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])
    elif num_of_axes == 3:
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animated_data = init(data, num_of_axes)
        colors = color_init(list(data.keys()))

        if num_of_axes == 2:
            for key, val in data.items():
                ax.scatter(val[0][0], val[1][0], c=colors[key], label=key)
        elif num_of_axes == 3:
            for key, val in data.items():
                ax.scatter(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_scatter, frames=len(data[first][0]), fargs=(ax, data, animated_data, colors, num_of_axes), interval=30)
        writergif = animation.PillowWriter()#fps=15)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        if num_of_axes == 2:
            for key, val in data.items():
                ax.scatter(val[0], val[1], label=key)
        elif num_of_axes == 3:
            for key, val in data.items():
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
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.autoscale(enable=True, axis='both')

    for key, val in data.items():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hexcolor = '#%02x%02x%02x'%(r, g, b)
        ax.stem(val[0], val[1], label=key, linefmt=hexcolor, markerfmt='D')
    plt.legend()

    if save_file == True:
        save(graph_name)
    else:
        plt.show()

################
# Main Program #
################

def main(argv=None):
    
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file (leave blank if parent directory is log/).')
    parser.add_argument('-f', '--file', action='store', type=str, help='Desired CSV file.')
    parser.add_argument('-c', '--column-headers', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line3d", "scatter", "scatter3d", "scatterh", "hist", "stem"]', default="line")
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for the generated graph.')
    parser.add_argument('-a', '--animated', action='store_true', help='Creates an animated graph when true (will be saved as a gif).')
    parser.add_argument('-l', '--live-view', action='store_true', help='Stream data from CSV files to Graph in real-time.')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file.')

    args = parser.parse_args()
    #print(args)

    if args.yaml is not None:
        import yaml
        data = dict()
        filepaths, headers = [], []

        try:
            filepath = os.path.join(currentFolder, args.yaml)
            with open(filepath, 'r') as f:
                yf = yaml.safe_load(f)
                is_live = yf['live'] if 'live' in yf.keys() else False
                for file in yf['files']:
                    key = list(file.keys())[0]
                    val = file[key]
                    if 'path' in val: 
                        path = os.path.join(log_path, val['path'])
                    else:
                        path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'])
                    filepath = get_file(path, val['name'])
                    print("Fetching %s"%(filepath))
                    if is_live == True:
                        filepaths.append(filepath)
                        headers.append(val['headers'])
                    else:            
                        data[key] = parse_csv(filepath, val['headers'])
                if 'z_label' in yf['labels']:
                    labels = [yf['labels']['x_label'], yf['labels']['y_label'], yf['labels']['z_label']]
                else:
                    labels = [yf['labels']['x_label'], yf['labels']['y_label']]
                title = yf['title'] if 'title' in yf.keys() else ''
                type = yf['type'] if 'type' in yf.keys() else 'line'
                animated = yf['animated'] if 'animated' in yf.keys() else False
                save_graph = yf['save'] if 'save' in yf.keys() else False

                if is_live == True:
                    live_plot(graph_name=title, labels=labels, filepaths=filepaths, headers=headers)
                else:
                    plot(graph_type=type, title=title, labels=labels, data=data, is_animated=animated, save_file=save_graph)
        except FileNotFoundError:
            print('%s does not exist.'%(filepath))
        except KeyError as k:
            print("KeyError: %s"%(k))
        except TypeError as t:
            print("One of the fields in your YAML file is not filled properly.\n%s"%(t))
    else:
        filename = args.file
        path_to_file = args.path if args.path is not None else ''
        filepath = get_file(os.path.join(log_path, path_to_file), filename) 
        title = args.title if args.title is not None else filename[:-4]
        labels = args.column_headers
        graph_type = args.graph_type
        animated = args.animated
        live = args.live_view
        values = parse_csv(filepath, labels)
        data = {filename: values}
        if live == True:
            live_plot(graph_name=title, labels=labels, filepaths=[filepath], headers=headers)
        else:
            plot(graph_type=graph_type, title=title, labels=labels, data=data, is_animated=animated, save_file=args.save)

if __name__ == "__main__":
    main()