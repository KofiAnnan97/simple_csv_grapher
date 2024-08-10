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

animated_data = dict()
colors = dict()

if not os.path.exists(animated_path):
    os.makedirs(animated_path)

###################
# General Methods #
###################

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
    if is_animated == True:
        print('Generating an animated gif may take a while based on the amount of data.')
    if graph_type == 'line':
        multi_2d_line(title,labels,data,is_animated,save_file)
    elif graph_type == 'line3d':
        multi_3d_line(title,labels,data,is_animated,save_file)
    elif graph_type == 'scatter':
        multi_scatter(title,labels,data,is_animated,save_file)
    elif graph_type == 'scatter3d':
        multi_3d_scatter(title,labels,data,is_animated,save_file)
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

def init(data):
    tmp = dict()
    for key,val in data.items():
        tmp[key] = list()
        for i in range(len(val)):
           tmp[key].append(list())
    return tmp

def color_init(data):
    tmp = dict()
    for key in data.keys():
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
            for j in range(len(val)):
                animated_data[key][j].append(val[j][i])  
            if num_of_axes == 2:
                lines = [ax.plot(x[:i], y[:i], c=colors[key]) for x, y in animated_data.values()]
            elif num_of_axes == 3:
                lines = [ax.plot(x[:i], y[:i], z[:i], c=colors[key]) for x, y, z in animated_data.values()]
        return lines
    except ValueError:
        """print("The number of labels and/or headers do not match the chosen graph type.")
        sys.exit()"""
        pass

def update_scatter(i, ax, data, animated_data, colors, num_of_axes):
    try:
        for key, val in data.items():
            for j in range(len(val)):
                animated_data[key][j].append(val[j][i])
            if num_of_axes == 2:  
                lines = [ax.scatter(x[:i], y[:i], c=colors[key]) for x, y in animated_data.values()]
            elif num_of_axes == 3:
                lines = [ax.scatter(x[:i], y[:i], z[:i], c=colors[key]) for x, y,z in animated_data.values()]        
        return lines
    except ValueError:
        """print("The number of labels and/or headers do not match the chosen graph type.")
        sys.exit(0)"""
        pass

####################
# Graphing Methods #
####################

def multi_2d_line(graph_name, labels, data, is_animated, save_file):
    fig, ax = plt.subplots()
    plt.title(graph_name)

    limits = get_limits(data)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])

    if is_animated == True:
        animated_data = init(data)
        colors = color_init(data)

        for key, val in data.items():
            ax.plot(val[0][0], val[1][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_lines, len(data[first][0]),fargs= (ax, data, animated_data, colors, 2), interval=100)
        writergif = animation.PillowWriter(fps=60)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        for key, val in data.items():
            ax.plot(val[0], val[1], label=key)
            plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_3d_line(graph_name,labels, data, is_animated, save_file):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animated_data = init(data)
        colors = color_init(data)

        for key, val in data.items():
            ax.plot(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_lines, len(data[first][0]), fargs=(ax, data, animated_data, colors, 3), interval=100)
        writergif = animation.PillowWriter(fps=60)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        for key, val in data.items():
            ax.plot(val[0], val[1], val[2], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_scatter(graph_name, labels, data, is_animated, save_file):
    fig, ax = plt.subplots()
    plt.title(graph_name)

    limits = get_limits(data)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel=labels[0], ylabel=labels[1])

    if is_animated == True:
        animated_data = init(data)
        colors = color_init(data)

        for key, val in data.items():
            ax.scatter(val[0][0], val[1][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_scatter, frames=len(data[first][0]), fargs=(ax, data, animated_data, colors, 2), interval=30)
        writergif = animation.PillowWriter()#fps=15)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        for key, val in data.items():
            ax.scatter(val[0], val[1], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()
    
def multi_3d_scatter(graph_name, labels, data, is_animated, save_file):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(graph_name)

    limits = get_limits(data)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])

    if is_animated == True:
        animated_data = init(data)
        colors = color_init(data)

        for key, val in data.items():
            ax.scatter(val[0][0], val[1][0], val[2][0], c=colors[key], label=key)
        plt.legend()

        first = list(data.keys())[0]
        anim = animation.FuncAnimation(fig, update_scatter, len(data[first][0]), fargs=(ax, data, animated_data, colors, 3), interval=100)
        writergif = animation.PillowWriter(fps=60)
        filename = create_filename(graph_name)
        filepath = os.path.join(animated_path, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        for key, val in data.items():
            ax.scatter(val[0], val[1], val[2], label=key)
        plt.legend()
        if save_file == True:
            save(graph_name)
        else:
            plt.show()

def multi_scatter_histogram(graph_name, labels, data, save_file):
    print("This figure may take a while to complete based on the size of data.")
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(graph_name)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set(aspect=1)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    xmin, xmax = float('inf'), -1*float('inf')
    ymin, ymax = float('inf'), -1*float('inf')

    for key, val in data.items():
        ax.scatter(val[0], val[1], label=key)

        binwidth = 0.25
        xmax = max(val[0]) if xmax < max(val[0]) else xmax
        xmin = min(val[0]) if xmin > min(val[0]) else xmin
        ymax = max(val[1]) if ymax < max(val[1]) else ymax
        ymin = min(val[1]) if ymin > min(val[1]) else ymin

        x_bins = np.arange(xmin, xmax + binwidth, binwidth)
        y_bins = np.arange(ymin, ymax + binwidth, binwidth)
   
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
    parser.add_argument('-c', '--column-names', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line3d", "scatter", "scatter3d", "scatterh", "hist", "stem"]', default="line")
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for generated graph.')
    parser.add_argument('-a', '--animated', action='store_true', help='Creates an animated graph when true (will be saved as a gif).')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file.')

    args = parser.parse_args()
    #print(args)

    if args.yaml is not None:
        import yaml
        data = dict()
        try:
            filepath = os.path.join(currentFolder, args.yaml)
            with open(filepath, 'r') as f:
                yf = yaml.safe_load(f)
                for file in yf['files']:
                    key = list(file.keys())[0]
                    val = file[key]
                    if 'path' in val: 
                        path = os.path.join(log_path, val['path'])
                    else:
                        path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'])
                    
                    if val['name'] == 'latest':
                        list_of_files = glob.glob(os.path.join(path, '*'), recursive=False)
                        list_of_files.sort()
                        filepath = max(list_of_files)
                    elif val['name'] == 'lastModified':
                        list_of_files = glob.glob(os.path.join(path, '*'), recursive=False)
                        filepath = max(list_of_files, key=os.path.getmtime)
                    else:
                        filepath = os.path.join(path, val['name'])
                    print("Fetching %s"%(filepath))
                    data[key] = parse_csv(filepath, val['headers'])

                if 'z_label' in yf['labels']:
                    labels = [yf['labels']['x_label'], yf['labels']['y_label'], yf['labels']['z_label']]
                else:
                    labels = [yf['labels']['x_label'], yf['labels']['y_label']]
                title = yf['title']
                type = yf['type']
                animated = yf['animated']
                save_graph = yf['save']
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
        filepath = os.path.join(log_path, path_to_file, filename) 
        title = args.title if args.title is not None else filename[:-4]
        col_names = args.column_names
        graph_type = args.graph_type
        animated = args.animated
        values = parse_csv(filepath, col_names)
        data = {filename: values}        
        plot(graph_type=graph_type, title=title, labels=col_names, data=data, is_animated=animated, save_file=args.save)

if __name__ == "__main__":
    main()