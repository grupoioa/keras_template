from config.MainConfig import getImageContourVisualizer
from inout.readData import readRawData
from inout.io_common_df import *
from visualization.TimeSeries import *

if __name__ == '__main__':

    opts_visual = getImageContourVisualizer()
    viz_mode = opts_visual['mode']
    data = readRawData(opts_visual)
    viz_obj = TimeSeries(data, opts_visual['display'])
    dataFrameSummary(data)

    if viz_mode == 'scatter':
        viz_obj.plotDfColumnsWithScatter(opts_visual['columns'], method=opts_visual['method'],
                                         legends=opts_visual['columns'])
