from pandas import DataFrame
from os.path import join
import pandas as pd

def readRawData(input_folder, file_name_x, file_name_y):
    '''
    Reads the raw data and returns X and Y seperated
    :param input_folder: Input Folder
    :param file_name_x: File name where X data is stored
    :param file_name_y: File name where Y data is stored
    :return:
    '''
    print(F'Reading {file_name_x} ....')
    X = pd.read_csv(join(input_folder,file_name_x))
    X['fecha'] = pd.to_datetime(X['fecha'],format='%Y-%m-%d %H:%M:%S')
    X = X.set_index(['fecha'])
    # dataFrameSummary(X)

    print(F'Reading {file_name_y} ....')
    Y = pd.read_csv(join(input_folder,file_name_y))
    Y['fecha'] = pd.to_datetime(Y['fecha'],format='%Y-%m-%d %H:%M:%S')
    Y = Y.set_index(['fecha'])
    # dataFrameSummary(Y)

    return X, Y




