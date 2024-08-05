import os
import argparse
import matplotlib.pyplot as plt
import csv

currentFolder = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(currentFolder, 'log/gps')

def get_col_idxs(col_names, header):
    tmp = []
    for col in col_names:
        idx = header.index(col)
        tmp.append(idx)
    return tmp

def parse_csv(filepath, col_names):
    with open(filepath, 'r') as cr:
        reader = csv.reader(cr)
        header = next(reader)
        col_idxs = get_col_idxs(col_names, header)

        vals = []
        for i in range(len(col_idxs)):
            vals.append(list())
        try:
            for row in reader:
                for i in range(len(col_idxs)):
                    idx = col_idxs[i]
                    vals[i].append(float(row[idx]))
        except Exception as e:
            pass

        return vals

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

def simple_2d_plot(graph_name, labels, x, y):
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(graph_name)
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.plot(x, y)
    plt.show()

def simple_scatter(graph_name, labels, x, y):
    fig, ax = plt.subplots()
    plt.title(graph_name)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.scatter(x, y)
    plt.show()

def main(argv=None):
    
    parser = argparse.ArgumentParser(prog='CSV Graphing', description='Simple program that graphs csv data.')
    parser.add_argument('-f', '--file', action='store', type=str, help='Choose file')
    parser.add_argument('-c', '--column-names', action='extend', nargs='+', type=str, help='Give desired column headers.')
    args = parser.parse_args()
    filename = args.file
    filepath = os.path.join(log_path, filename)
    col_names = args.column_names

    values = parse_csv(filepath, col_names)
    #convert_gps_to_xy(filename, values)
    
    '''
    Graphing Methodologies
    '''
    #simple_2d_plot(filename, col_names, values[0], values[1])
    #simple_scatter(filename, col_names, values[0], values[1])

if __name__ == "__main__":
    main()