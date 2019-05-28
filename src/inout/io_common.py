from os.path import join
from os import walk, listdir
import os
import SimpleITK as sitk
from pandas import Series


def saveDictionaryToFile(my_dict, file_name):
    '''THis function will write the values of a dictionary into a csv, BUT it will also
    append the mean value as the last row'''
    data= Series(my_dict, index=my_dict.keys())
    mean_value = data.mean()
    data['AVG'] = mean_value
    data.sort_index(axis=0,inplace=True)
    data.to_csv(file_name)


def saveMultipleDictionaryToFile(all_dicts, file_name, names):
    '''THis function will write the values of a dictionary into a csv, BUT it will also
    append the mean value as the last row'''
    for ii, my_dict in enumerate(all_dicts):
        if ii == 0:
            data= Series(my_dict, index=my_dict.keys())

    mean_value = data.mean()
    data['AVG'] = mean_value
    data.sort_index(axis=0,inplace=True)
    data.to_csv(file_name)

def filterColumns(df, columns, mode='keep'):
    '''
    Filter some columns in a dataframe
    :param df:
    :param columns:
    :param mode: 'keep', 'remove' Keeps or removes the specified columns
    :return:
    '''
    if mode == 'keep':
        print(F'Filtering columns, only keeping {columns}')
        return df.loc[columns]
    else:
        print(F'Filtering columns, removing keeping {columns}')
        return df.drop(columns)


def dataFrameSummary(df):
    ''' Displays a summary of a dataframe '''
    print(F'\n*******  Header:\n{df.head()}') # Show first few rows
    print(F'\n*******  Columns :\n{df.columns}')
    print(F'\n*******  Columns Types :\n{df.dtypes}')
    print(F'\n*******  Shape:\n{df.shape}')


def createFolder(folder):
    ''' Simply wrapper to create a folder if it doesn't exist '''
    if not(os.path.exists(folder)):
        os.makedirs(folder)
