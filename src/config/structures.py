from enum import Enum

# Enumerations
class GRAPH_TYPES(Enum):
    LINE = "line"
    LINE3D = "line3d" 
    SCATTER = "scatter"
    SCATTER3D = "scatter3d"
    SCATTERH = "scatterh"
    HISTOGRAM = "hist"
    STEMPLOT = "stem"
    LINE_XX = "line_xx"
    LINE_YY = "line_yy"
    SCATTER_XX = "scatter_xx"
    SCATTER_YY = "scatter_yy"
    PERF2D = "perf2d"
    PERF3D = "perf3d"

# C++ - like Structures
class Parameters:
    __slots__ = ("title", "labels", "data", "graph_type", "save_file", "is_animated", "is_live", "metrics_config")
    def __init__(self) -> None:
        self.title = ''
        self.labels = list()
        self.data = dict()
        self.graph_type = GRAPH_TYPES.LINE.value
        self.save_file = False
        self.is_animated = False
        self.is_live = False
        self.metrics_config = dict()

class SavePaths:
    __slots__ = {"animated_dir", "metrics_dir", "static_dir"}
    def __init__(self) -> None:
        self.animated_dir = ''
        self.metrics_dir = ''
        self.static_dir = ''

# Number of axes used based on graph type 
graph_axes = {
    GRAPH_TYPES.LINE.value: 2,
    GRAPH_TYPES.LINE3D.value: 3, 
    GRAPH_TYPES.SCATTER.value: 2,
    GRAPH_TYPES.SCATTER3D.value: 3,
    GRAPH_TYPES.SCATTERH.value: 2,
    GRAPH_TYPES.HISTOGRAM.value: 2,
    GRAPH_TYPES.STEMPLOT.value: 2,
    GRAPH_TYPES.LINE_YY.value: 2,
    GRAPH_TYPES.SCATTER_YY.value: 2,
    GRAPH_TYPES.PERF2D.value: 2,
    GRAPH_TYPES.PERF3D.value: 3
}