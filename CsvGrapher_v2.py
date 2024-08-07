import os
import argparse
import matplotlib.pyplot as plt
import csv
import sys
import glob

currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')

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
        print('%s does not exist. Check that path and name is correct for the field "files" in your YAML file.'%(filepath))
        sys.exit(0)
    
    return vals

def multi_2d_line(graph_name, labels, data, save_file):
    fig, ax = plt.subplots()
    plt.title(graph_name)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.autoscale(enable=True, axis='both')

    for key, val in data.items():
        ax.plot(val[0], val[1], label=key)
    plt.legend()
    if save_file == True:
        save_plot(graph_name)
    else:
        plt.show()

def multi_3d_line(graph_name,labels, data, save_file):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(graph_name)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    for key, val in data.items():
        ax.plot(val[0], val[1], val[2], label=key)
    
    ax.set_autoscalex_on = True
    ax.set_autoscaley_on = True
    plt.legend()
    if save_file == True:
        save_plot(graph_name)
    else:
        plt.show()

def multi_scatter(graph_name, labels, data, save_file):
    fig, ax = plt.subplots()
    plt.title(graph_name)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.autoscale(enable=True, axis='both')

    for key, val in data.items():
        ax.scatter(val[0], val[1], label=key)
    plt.legend()
    if save_file == True:
        save_plot(graph_name[:-4])
    else:
        plt.show()

def save_plot(title):
    print(title)
    import re
    from datetime import datetime
    graphs_path = os.path.join(log_path, 'graphs')
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    stamp = datetime.now().isoformat('_', timespec='seconds')
    filename = '%s_%s'%(stamp, title)
    filename = re.sub('\-|\:|\s+', '_', filename)
    filepath = os.path.join(graphs_path, filename)
    plt.savefig(filepath)
    print("%s.png has been created."%(filepath))

def main(argv=None):
    
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-f', '--file', action='store', type=str, help='Desired CSV file.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file (leave blank if parent directory is log/).')
    parser.add_argument('-c', '--column-names', action='extend', nargs='+', type=str, help='Give desired column headers (leave spaces between each header).')
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "line3d", "scatter"]', default="line")
    parser.add_argument('-t', '--title', action='store', type=str, help='Provide title for generated graph')
    parser.add_argument('-s', '--save', action='store_true', help='Save graph.')
    parser.add_argument('-y', '--yaml', action='store', type=str, help='Generate graph via yaml config file')

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
                    if val['name'] == 'latest':
                        if 'path' in val: 
                            path = os.path.join(log_path, val['path'], '*')
                        else:
                            path = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'], '*')
                        list_of_files = glob.glob(path)
                        filepath = max(list_of_files, key=os.path.getctime)
                        print("Fetching %s"%(filepath))
                    else:
                        if 'path' in val:
                            filepath = os.path.join(log_path, val['path'], val['name'])
                        else:
                            filepath = os.path.join(log_path, val['bcn_type'], val['comm_port_no'], val['folder'], val['name'])
                    data[key] = parse_csv(filepath, val['headers'])
                if 'z_label' in yf['labels']: 
                    labels = [yf['labels']['x_label'], yf['labels']['y_label'], yf['labels']['z_label']]
                else:
                    labels = [yf['labels']['x_label'], yf['labels']['y_label']]
                title = yf['title']
                type = yf['type']
                save_graph = yf['save']
                if type == 'line':
                    multi_2d_line(title,labels,data,save_graph)
                elif type == 'line3d':
                    multi_3d_line(title,labels,data,save_graph)
                elif type == 'scatter':
                    multi_scatter(title,labels,data,save_graph)
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
        col_names = args.column_names
        graph_type = args.graph_type
        title = args.title if args.title is not None else filename

        values = parse_csv(filepath, col_names)
        data = {filename: values}
        
        if graph_type == 'line':
            multi_2d_line(title,col_names,data,args.save)
        elif graph_type == 'line3d':
            multi_3d_line(title,col_names,data,args.save)
        elif graph_type == 'scatter':
            multi_scatter(title,col_names,data,args.save)
        else:
            print("Unrecognized graph type: %s"%(graph_type))

if __name__ == "__main__":
    main()