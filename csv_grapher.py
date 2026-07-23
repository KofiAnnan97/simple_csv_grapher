import os
import argparse
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.animation as animation
from typing import Dict, List, Any, Tuple

# Project files
from  src.util.parsers import CSVParser, DataParser
from src.util.file_helper import create_filename
from src.util.plot_helper import get_limits, color_dict
from src.metrics.metrics import PerformanceMetrics, ATE, METRIC_TYPES
from src.config.structures import GRAPH_TYPES, Parameters, SavePaths, graph_axes
from src.plotting.live_view import live_plot
from src.plotting.methods import second_axis, multi_line, multi_scatter, multi_scatter_histogram, multi_histogram, multi_stem, position_perf

# Data configuration
curr_dir = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(curr_dir, 'logs')
graphs_path = os.path.join(curr_dir, 'graphs')
animated_path = os.path.join(graphs_path, 'animated')
metrics_path = os.path.join(graphs_path, 'metrics')
static_path = os.path.join(graphs_path, 'static')

if not os.path.exists(animated_path):
    os.makedirs(animated_path)

if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

if not os.path.exists(static_path):
    os.makedirs(static_path)

def create_plot(params: Parameters, save_paths: SavePaths) -> None:
    if params.is_animated:
        print('Generating an animated gif may take some time (based on the quantity of data).')
    if params.graph_type in [GRAPH_TYPES.LINE.value,GRAPH_TYPES.LINE3D.value]:
        multi_line(params, save_paths)
    elif params.graph_type in [GRAPH_TYPES.LINE_XX.value,GRAPH_TYPES.LINE_YY.value,GRAPH_TYPES.SCATTER_XX.value,GRAPH_TYPES.SCATTER_YY.value]:
        second_axis(params, save_paths)
    elif params.graph_type in [GRAPH_TYPES.SCATTER.value,GRAPH_TYPES.SCATTER3D.value]:
        multi_scatter(params, save_paths)
    elif params.graph_type == GRAPH_TYPES.SCATTERH.value:
        multi_scatter_histogram(params, save_paths)
    elif params.graph_type == GRAPH_TYPES.HISTOGRAM.value:
        multi_histogram(params, save_paths)
    elif params.graph_type == GRAPH_TYPES.STEMPLOT.value:
        multi_stem(params, save_paths)
    elif params.graph_type in [GRAPH_TYPES.PERF2D.value, GRAPH_TYPES.PERF3D.value]:
        position_perf(params, save_paths)
    else:
        print("Unrecognized graph type: %s"%(params.graph_type))
        sys.exit(0)

def show_metric(metric_name: str, metric_type: str, ground_truth: Any, p: Parameters, sp: SavePaths) -> None:
    if metric_type == METRIC_TYPES.ATE.value:
        ATE.show_ate_results(metric_name, ground_truth, graphs_path, p, sp.metrics_dir)

################
# Main Program #
################

def main():
    save_paths = SavePaths()
    save_paths.animated_dir = animated_path
    save_paths.metrics_dir = metrics_path
    save_paths.static_dir = static_path

    parser = argparse.ArgumentParser(prog='CSV Graphing', description='A simple program that graphs data from csv files.')
    parser.add_argument('-p', '--path', action='store', type=str, help='Path to desired file from current working directory (leave blank if parent directory is logs/).')
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
                filepath = os.path.join(curr_dir, "%s.yaml"%(args.yaml))
            else:
                filepath = os.path.join(curr_dir, args.yaml)
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
                        path = os.path.join(curr_dir, val['path'])
                    else:
                        path = os.path.join(curr_dir, val['bcn_type'], val['comm_port_no'], val['folder'])
                    filepath = DataParser.get_file(path, val['name'])
                    group_by = val['group_by'] if 'group_by' in val else None
                    print("Fetched %s"%(filepath))
                    if params.is_live:
                        params.data['filepaths'].append(filepath)
                        params.data['headers'].append(val['headers'])
                        params.data['names'].append(key)
                        params.data['group_by'].append(group_by)
                    else:
                        csv_data = CSVParser.parse_csv(filepath, val['headers'], group_by) # type: ignore
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
                    create_plot(params, save_paths)

                if 'metrics' in yf:
                    for metric in yf['metrics']:
                        key = list(metric.keys())[0]
                        definition = metric[key]
                        params.metrics_config[key] = definition    
                    for key, val in params.metrics_config.items():
                        show_metric(key, val['type'], val['ground_truth'], params, save_paths)

        except FileNotFoundError:
            print('YAML file: %s does not exist.'%(filepath))
        except KeyError as k:
            print("KeyError: %s"%(k))
        except TypeError as t:
            print("One of the fields in your YAML file is not filled properly.\n%s"%(t))
    else:
        filename = args.file
        path_to_file = os.path.join(curr_dir, args.path) if args.path is not None else log_path
        filepath = DataParser.get_file(path_to_file, filename) 
        params.title = args.title if args.title is not None else filename[:-4]
        params.graph_type = args.graph_type if args.graph_type is not None else GRAPH_TYPES.LINE.value
        group_by = args.group_by
        params.labels = args.column_headers if args.column_headers is not None else []
        params.is_live = args.live_view if args.live_view is not None else False
        if params.is_live:
            params.data = {'filepaths':[filepath], 'headers':[params.labels], 'names':[filename[:-4]], 'group_by':[group_by]}
            live_plot(metadata=params.data,title=params.title,labels=params.labels,graph_type=params.graph_type)
        else:
            params.save_file = args.save if args.save is not None else False
            params.is_animated = args.animated if args.animated is not None else False
            values = CSVParser.parse_csv(filepath, args.column_headers, group_by)
            params.data = {filename: values} if group_by is None else values # type: ignore
            create_plot(params, save_paths) 

if __name__ == "__main__":
    main()