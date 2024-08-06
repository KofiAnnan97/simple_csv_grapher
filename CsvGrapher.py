import os
import argparse
import matplotlib.pyplot as plt
import csv
import sys

currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log')

class CsvParser:
    def __init__(self):
        self.data = dict()
        self.xy_labels = ['x', 'y']
        self.title = 'test'
        self.graph_type = 'line' #'' 
        self.methods = GraphingMethods()

    def get_col_idxs(self, col_names, header):
        tmp = []
        for col in col_names:
            idx = header.index(col)
            tmp.append(idx)
        return tmp

    def parse_csv(self, filepath, col_names):
        with open(filepath, 'r') as cr:
            reader = csv.reader(cr)
            header = next(reader)
            col_idxs = self.get_col_idxs(col_names, header)

            vals = []
            for i in range(len(col_idxs)):
                vals.append(list())
            try:
                for row in reader:
                    for i in range(len(col_idxs)):
                        idx = col_idxs[i]
                        vals[i].append(float(row[idx]))
            except Exception as e:
                print(e)
            return vals
    
    def cli_menu(self,choice):
        if choice == '1':
            filename = input('CSV Filename: ')
            filepath = os.path.join(log_path, filename)
            headers = input('Give desired column headers (separated by spaces): ')
            col_names = headers.split(' ')
            self.data[filename] = self.parse_csv(filepath, col_names)
            print(list(self.data.keys()))
        elif choice == '2':
            print("Current data:", list(self.data.keys()))
            file = input("Remove one of the following csv files: ")
            try:
                del self.data[file]
            except KeyError:
                print("Could not find data for: %s"%(file))
        elif choice == '3':
            x_label = input("X Label: ")
            y_label = input("Y Label: ")
            self.xy_labels = [x_label, y_label]
        elif choice == '4':
            self.title = input('Set Title: ')
        elif choice == '5':
            type = input('Choose one of the following ["line", "scatter"]: ')
            if type == "line":
                self.graph_type = type
            elif type == "scatter":
                self.graph_type = type
            print('Current Graph Type: %s'%(self.graph_type))
        elif choice == '6':
            save_input = input("Save graph as file (y/n): ")
            save_file = True if save_input == 'y' else False
            if self.graph_type == "line":
                self.methods.multi_2d_plot(self.title, self.xy_labels, self.data, save_file)
            elif self.graph_type == "scatter":
                self.methods.multi_scatter(self.title, self.xy_labels, self.data, save_file)
            else:
                print("Unrecognized graph type: %s"%(self.graph_type))
        elif choice == '7':
            sys.exit(0)
        else:
            print("Invalid option: %s"%(choice))

"""def convert_gps_to_xy(file, values):
    import utm
    tmp = []
    for i in range(len(values)):
        tmp.append(list())
    for j in range(len(values[0])):
        data = utm.from_latlon(values[0][j], values[1][j])
        tmp[0].append(data[0])
        tmp[1].append(data[1])
    filename = 'xyz' + file[3:]
    filepath = os.path.join(log_path, filename)
    with open(filepath, 'w') as cw:
        writer = csv.writer(cw, delimiter=',')
        writer.writerow(["x","y"])
        #print(tmp)
        for k in range(len(tmp[0])):
            writer.writerow([tmp[0][k], tmp[1][k]])"""

class GraphingMethods:
    def __init__(self):
        pass

    def simple_2d_plot(self, graph_name, labels, x, y):
        fig, ax = plt.subplots()
        plt.title(graph_name)
        
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.plot(x, y)
        plt.show()

    def multi_2d_plot(self, graph_name, labels, data, save_file):
        fig, ax = plt.subplots()
        plt.title(graph_name)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        for key, val in data.items():
            ax.plot(val[0], val[1], label=key)
        plt.legend()
        if save_file == True:
            self.save_plot(graph_name)
        else:
            plt.show()

    def simple_scatter(self, graph_name, labels, x, y):
        fig, ax = plt.subplots()
        plt.title(graph_name)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.scatter(x, y)
        plt.show()

    def multi_scatter(self, graph_name, labels, data, save_file):
        fig, ax = plt.subplots()
        plt.title(graph_name)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        for key, val in data.items():
            ax.scatter(val[0], val[1], label=key)
        plt.legend()
        if save_file == True:
            self.save_plot(graph_name)
        else:
            plt.show()

    def save_plot(self, title):
        import re
        from datetime import datetime
        graphs_path = os.path.join(log_path, 'graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        stamp = datetime.now().isoformat('_', timespec='seconds')
        filename = '%s_%s'%(stamp, title)
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r':', '_', filename)
        filename = re.sub(r'-', '_', filename)
        filepath = os.path.join(graphs_path, filename)
        plt.savefig(filepath)
        print("%s.png has been created."%(filepath))

def main(argv=None):
    
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='Simple program that graphs csv data.')
    parser.add_argument('-f', '--file', action='store', type=str, help='Choose file')
    parser.add_argument('-c', '--column-names', action='extend', nargs='+', type=str, help='Give desired column headers.')
    parser.add_argument('-m', '--menu', action='store_true', help="Bring up command line menu")
    parser.add_argument('-g', '--graph-type', action='store', help='Choose one of the following ["line", "scatter"]', default="line")

    args = parser.parse_args()
    #print(args)
    csv_parser = CsvParser()
    methods = GraphingMethods()

    if args.menu == True:
        while True:
            print("""
        Simple CSV Grapher Menu:
        1.) Import Data from CSV File
        2.) Remove Data
        3.) Set XY Labels
        4.) Set Title
        5.) Choose Graph
        6.) Plot Data
        7.) Quit
        """)
            option = input("Option: ")
            csv_parser.cli_menu(option)
    else:
        filename = args.file
        filepath = os.path.join(log_path, filename)
        col_names = args.column_names
        graph_type = args.graph_type

        values = csv_parser.parse_csv(filepath, col_names)
        #convert_gps_to_xy(filename, values)
        
        if graph_type == 'line':
            methods.simple_2d_plot(filename, col_names, values[0], values[1])
        elif graph_type == 'scatter':
            methods.simple_scatter(filename, col_names, values[0], values[1])
        else:
            print("Unrecognized graph type: %s"%(graph_type))

if __name__ == "__main__":
    main()