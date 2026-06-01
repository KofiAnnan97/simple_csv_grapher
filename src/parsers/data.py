from glob import glob
import os

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