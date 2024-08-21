import os
import csv
import sys
import glob

currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')

VALID_GRAPH_TYPES = ["line", "line3d", "scatter", "scatter3d", "scatterh", "hist", "stem"]

class DataConfig:
    def __init__(self):
        self.title = ''
        self.labels = []
        self.data = {}
        self.graph_type = 'line'
        self.live_view = False
        self.is_animated = False
        self.can_save = False
        self.save_file = False
        self.init_path = log_path

    def can_plot(self):
        return ((self.title != '') and 
                (len(self.labels) > 0) and 
                (len(self.data) > 0))

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

    def get_col_idxs(self, col_names, header):
        tmp = []
        for col in col_names:
            idx = header.index(col)
            tmp.append(idx)
        return tmp

    def parse_csv(self, filepath, col_names):
        vals = []
        try:
            with open(filepath, 'r') as cr:
                reader = csv.reader(cr)
                header = next(reader)
                col_idxs = self.get_col_idxs(col_names, header)

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

    def set_title(self, name):
        self.title = name

    def set_labels(self, data):
        self.labels = data

    def add_data(self, path, filename, headers, key=None):
        filepath = self.get_file(path, filename)
        filedata = self.parse_csv(filepath, headers)
        if key is not None:
            self.data[key] = filedata
        else:
            self.data[filename[:-4]] = filedata
        pass

    def remove_data(self, key):
        try:
            del self.data[key]
        except KeyError:
            print("Could not find data for: %s"%(key))

    def show_data(self):
        print(self.data)

    def set_graph_type(self, g_type):
        if g_type in VALID_GRAPH_TYPES:
            self.graph_type = g_type
        else:
            print("Unrecognized graph type: %s"%(g_type))

    def set_live_view(self, state):
        self.live_view = state if type(state) == bool else False

    def set_animation(self, state):
        self.is_animated = state if type(state) == bool else False

    def set_save_flag(self, state):
        self.can_save = state if type(state) == bool else False
