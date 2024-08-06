# Simple CSV Grapher 
**When running the following script make sure that your ```log/``` folder is at the same directory level as this script.**

![Example Graph](./example.png)

## Table of Contents
- [Prerequistes](#python3-prerequistes)
- [Usage](#usage)
    - [Command line](#command-line)
    - [YAML Config File](#yaml-config-file)

## Python3 Prerequistes:
```bash
$ pip install matplotlib pyyaml
```
## Usage
This script can be run in two distinct manners. If you want to grap data for a single csv file command line may be the quickest option. Other wise if you want to graph multiple csv files then the yaml configuration is better suited.  
### Command line
This method allows the user to plot and save a png for based on a specified csv file. It only support one file at a time. 
```bash
usage: CSV Graphing [-h] [-f FILE] [-p PATH] [-c COLUMN_NAMES [COLUMN_NAMES ...]] [-g GRAPH_TYPE] [-t TITLE] [-s] [-y YAML]

Simple program that graphs csv data.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Choose file
  -p PATH, --path PATH  Path to file (leave blank if in log directory)
  -c COLUMN_NAMES [COLUMN_NAMES ...], --column-names COLUMN_NAMES [COLUMN_NAMES ...]
                        Give desired column headers.
  -g GRAPH_TYPE, --graph-type GRAPH_TYPE
                        Choose one of the following ["line", "scatter"]
  -t TITLE, --title TITLE
                        Provide title for generated graph
  -s, --save            Save file based on csv file.
  -y YAML, --yaml YAML  Generate graph via yaml file
```
Go to ``` python3 CsvGrapher_v2.py - h``` for up to date descriptions on how to use the command line. 

An example of how to use the command line below.
```bash
$ python3 CsvGrapher_v2.py -p "path/to/file" -f filename.csv -c header_1 header_2 -g "line" -t "Example Line Graph" -s
```
### YAML Config File
If you want to use multiple csv files the yaml configuration method is the best option. As long as the yaml file is at the same level as this script you can run the ```-y``` to run the configuration of a yaml file. The following fields are used when writing a yaml file
- ```files``` is the keyword used to find gather all the files that will be used to generate a graph. 
- ```path``` is the path where to find log file is inside the ```log/``` directory. If the file is in this directory put empty single quotes ```''``` in this field
- ```name``` is the name of the csv file being used. Using the word latest will grab the last generated file. 
- ```headers``` correlates to the column names that will be searched when retrieving data. Should be provided as a list. 
- ```labels``` correlates to the x and y labels (only support s one of each).  
- ```title``` is the title of the generated graph. 
- ```type``` is the type of graph that this script supports (Currently a line graph or scatter plot).
- ```save``` is a boolean value that determines whether the graph is saved (when true) or just plotted (when false).

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
title: Example 
type: line
save: false
```

An example of how to use the yaml config file below.
```bash
$ python3 CsvGrapher_v2.py -y test.yaml
```