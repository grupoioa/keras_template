from config.MainConfig import getPreprocessingConfig
from inout.readData import readRawData
from inout.io_common_df import *
from preproc.preprocDataFrames import *
from visualization.TimeSeries import *
from os.path import join

if __name__ == '__main__':

    preproc_options = getPreprocessingConfig()

    # This section is specific to our problem
    input_folder= preproc_options['input_folder']
    file_name_x = preproc_options['file_names_x']
    file_name_y = preproc_options['file_names_y']

    output_folder= preproc_options['output_folder']
    norm_data = preproc_options['normalize']

    data_x, data_y = readRawData(input_folder, file_name_x, file_name_y)

    if norm_data:
        # Normalize data
        data_x_norm = normalizeZeroToOne(data_x)
        data_y_norm = normalizeZeroToOne(data_y)

    if preproc_options['display']:
        # If you want to filter data
        plot_columns = preproc_options['plot_columns']
        save_fig = preproc_options['save_images']
        viz_obj = TimeSeries(data_x_norm, disp=True, output_folder=output_folder, save_fig=save_fig)
        viz_obj.plotDfColumnsWithScatter(plot_columns, xcol='index')

    # Save preprocessed data
    data_x_norm.to_csv(join(output_folder,file_name_x))
    data_y_norm.to_csv(join(output_folder,file_name_y))
