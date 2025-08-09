from glob import glob
import os
import csv
from copy import copy

from typing import Dict, List, Any, Tuple

class DataParser:
    @staticmethod
    def get_file(path: str, filename: str) -> str:
        if filename == 'latest':
            list_of_files = glob(os.path.join(path, '*'), recursive=False)
            list_of_files.sort()
            filepath = max(list_of_files)
        elif filename == 'lastModified':
            list_of_files = glob(os.path.join(path, '*'), recursive=False)
            filepath = max(list_of_files, key=os.path.getmtime)
        else:
            filepath = os.path.join(path, filename)
        return filepath

class CSVParser:
    @staticmethod
    def get_col_idxs(col_names: List[str], headers: List[str]) -> List[int]:
        tmp = []
        for col in col_names:
            try:
                idx = headers.index(col)
                tmp.append(idx)
            except ValueError:
                raise ValueError('Column header \'%s\' is not in the csv file.'%(col))
        return tmp
        
    @staticmethod
    def parse_csv(filepath: str, col_names: List[str], group_by: str=''):
        vals = []
        groups = dict()
        tmp_cols = copy(col_names)
        try:
            with open(filepath, 'r') as cr:
                if group_by != '':
                    reader = csv.reader(cr)
                    headers = next(reader)
                    tmp_cols.insert(0, group_by)
                    col_idxs = CSVParser.get_col_idxs(tmp_cols, headers)
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
                    col_idxs = CSVParser.get_col_idxs(tmp_cols, headers)
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
        if (isinstance(vals, list) and len(vals) == 0) or (isinstance(vals, dict) and len(vals.keys()) == 0):
            raise Exception('%s has no data.'%(filepath))
        return vals
