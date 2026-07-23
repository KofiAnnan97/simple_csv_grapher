from math import floor
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os, sys
from typing import Dict, List, Any, Tuple

from src.config.structures import SavePaths

def create_filename(title: str) -> str:
    import re
    from datetime import datetime
    stamp = datetime.now().isoformat('_', timespec='seconds')
    filename = '%s_%s'%(stamp, title)
    filename = re.sub('\-|\:|\s+', '_', filename)
    return filename

def save(title: str,save_paths: SavePaths, anim:Any=None) -> None:
    filename = create_filename(title)
    if anim is not None:
        writergif = animation.PillowWriter(fps=60)
        filepath = os.path.join(save_paths.animated_dir, "%s.gif"%(filename))
        anim.save(filepath, writer=writergif)
        print('%s saved.'%(filepath))
    else:
        filepath = os.path.join(save_paths.static_dir, filename)
        plt.savefig(filepath)
        print("%s.png has been created."%(filepath))