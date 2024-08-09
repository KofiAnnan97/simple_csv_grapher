# Simple CSV Grapher 
**When running the following script make sure that your ```log/``` folder is at the same directory level as this script.**

![Example Line Graph](./example_plots/line.png)

## Table of Contents
- [Python3 Prerequistes](#python3-prerequistes)
- [Usage](#usage)
    - [Command line](#command-line)
    - [YAML Config File](#yaml-config-file)

## Python3 Prerequistes:
```bash
$ pip install matplotlib pyyaml numpy
```
## Usage
This script can be run in two distinct manners (command line and configuration file). To graph data from a single csv command line may be the quickest option, though the other option still works. For graph data from multiple csv files the yaml configuration is only option.  
### Command line
This method allows the user to plot and save a .png file based on a specified csv file. It only support one file at a time. 
```
usage: CSV Graphing [-h] [-f FILE] [-p PATH] [-c COLUMN_NAMES [COLUMN_NAMES ...]] [-g GRAPH_TYPE] [-t TITLE] [-s] [-y YAML]

A simple program that graphs data from csv files.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Desired CSV file.
  -p PATH, --path PATH  Path to desired file (leave blank if parent directory is log/).
  -c COLUMN_NAMES [COLUMN_NAMES ...], --column-names COLUMN_NAMES [COLUMN_NAMES ...]
                        Give desired column headers (leave spaces between each header).
  -g GRAPH_TYPE, --graph-type GRAPH_TYPE
                        Choose one of the following ["line", "line3d", "scatter", "scatter3d", "scatterh", "hist", "stem"]
  -t TITLE, --title TITLE
                        Provide title for generated graph.
  -s, --save            Save graph.
  -y YAML, --yaml YAML  Generate graph via yaml config file.
```
Go to ``` python3 CsvGrapher_v2.py -h``` for an up to date description on how to use the command line. 

Here's an example of how to use the command line.
```bash
$ python3 CsvGrapher_v2.py -p "path/to/file" -f filename.csv -c header_1 header_2 -g "line" -t "Example Line Graph" -s
```
### YAML Config File
Graphing multiple csv files requires the use of the yaml configuration method. To use this method run ```-y``` with the name of the yaml file (makes sure the yaml file is at the same level as this script). The following fields are used when writing a yaml file.
- **```files```** is the keyword used to find all the files that will be used to generate a graph. 
- **```path```** is the path to find the csv file from within the ```log/``` directory. If the file is directly in the ```log/``` directory put empty single quotes ```''``` in this field.
- **```name```** is the csv filename. Using the keyword ```latest``` will grab the last generated file alphabetically. Using the keyword ```lastModified``` will grab the last modified file.  
- **```headers```** correlates to the column names that will be searched when retrieving data. Should be provided as a list of strings. 
- **```labels```** correlates to the x, y and z labels (only supports one of each). The z label is option.  
- **```title```** is the title of the generated graph. 
- **```type```** determines what type of graph will be generated. Currently supports: line graphs (2D and 3D), scatter plot (2D and 3D), scatter plot with histograms, histogram and stem plot.
- **```save```** is a boolean value that determines whether the graph is saved (when true) or just plotted (when false).

Here's an example of how a yaml file can be made.

```yaml
files:
  - file1:
      path: path/to/file1
      name: file1.csv
      headers: [col1, col2]
  - file2:
      path: path/to/file2
      name: file2.csv
      headers: [col1, col2]
labels:
  x_label: col1
  y_label: col2
title: Example Scatter Plot
type: scatter
save: false
```

Here's an example of how to run the script with the  yaml config file.
```bash
$ python3 CsvGrapher_v2.py -y example.yaml
``` 
Example output:

![Example Scatter Plot](./example_plots//scatter.png)