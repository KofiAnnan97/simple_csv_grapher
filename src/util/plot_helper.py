import random
from typing import Dict, List, Any, Tuple

def color_dict(data: List[str]) -> Dict[Any, Any]:
    tmp = dict()
    for key in data:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hexcolor = '#%02x%02x%02x'%(r, g, b)
        tmp[key] = hexcolor
    return tmp

def get_limits(data: Dict[Any, Any]) -> List[Any]:
    tmp = []
    for val in list(data.values())[0]:
        tmp.append([float('inf'), -1*float('inf')])
    for val in data.values():
        for i in range(len(val)):
            tmp[i][0] = min(val[i]) if tmp[i][0] > min(val[i]) else tmp[i][0]
            tmp[i][1] = max(val[i]) if tmp[i][1] < max(val[i]) else tmp[i][1]
    for lim in tmp:
        diff = abs(lim[1] - lim[0])
        offset = 0.15*diff if diff > 0 else 0.5
        lim[0] = lim[0] - offset
        lim[1] = lim[1] + offset

    return tmp