import numpy as np
import pandas as pd
from pandas import DataFrame

def normalizeZeroToOne(df, columns='all'):
    '''Normalizes de data from 0 to 1'''

    if columns == 'all':
        x = df
    else:
        x = df[columns]

    r = (x.max() - x.min())
    if np.any(r == 0):
        print(F'Warning!!!  These columns only have a single value (removing them): {r[r==0]}')

    x_norm = (x - x.min()) / r
    # x_norm.dropna(axis=1,inplace=True) # Remove empty columns
    x_norm.fillna(0)
    return x_norm
