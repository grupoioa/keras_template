
def getPreprocessingConfig():
    '''These method creates configuration parameters for the preprocessing code.'''
    cur_config = {
        'input_folder': '../example_data',  # Where your data is
        'output_folder': '../dataPreproc',  # Where to save preprocessed data
        'normalize': True,  # Do you want to normalize the data
        'file_names_x': 'AJM_O3.csv',  # X data file
        'file_names_y': 'AJM_O3_pred.csv',  # Y data file
        'display': False,  #  Do you want to plot intermediate results
        'save_images': False,  #  If disp=True you can decide to save the images
        # plot_columns is kind of specific for datafrrames, you normally wont need it
        'plot_columns':  ['weekday','sinWeekday','cont_pmco_ajm','cont_otres_ajm','year',
                                 'month','sinMonth','day','sinDay','valLab','U10','U10']
        }

    return cur_config


def getTrainingConfig():
    '''These method creates configuration parameters for the preprocessing code.'''
    cur_config = {
        'input_folder': '../dataPreproc',
        'output_folder': '../output',
        'model_name': 'default',
        'epochs': 1000,
        }

    return cur_config


def getImageContourVisualizer():
    '''These method creates configuration parameters for making visualizations'''
    cur_config = {
        'input_folder': '../dataDebugging',
        # 'input_folder': '../data',
        'output_folder': '../output/images/raw',
        'file_names_x': ['AJM_O3.csv'],
        'columns': ['weekday','sinWeekday','cont_pmco_ajm','cont_otres_ajm','year',
                    'month','sinMonth','day','sinDay','valLab','U10','U10'],
        'mode': 'scatter',
        'method': 'single', # 'single': Separated windows 'merged': all in same figure  'subplot' Single image multiple plots
        'display': True
        }

    # for x in range(1,99):
    #     cur_config['columns'].append(F'U10.{x}')

    print(cur_config['columns'])


    return cur_config
