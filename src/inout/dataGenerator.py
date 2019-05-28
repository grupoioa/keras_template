import numpy as np
from inout.readData import *

def dataGenerator(options, example_ids, batch_size=1):

    """
    Generator to yield inputs and their labels in batches.
    """
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(example_ids)))
    input_folder = options['input_folder']
    file_name_x = 'AJM_O3.csv'
    file_name_y = 'AJM_O3_pred.csv'
    print(F'Reading {file_name_x} ....')
    orig_x = pd.read_csv(join(input_folder,file_name_x))
    orig_x['fecha'] = pd.to_datetime(orig_x['fecha'],format='%Y-%m-%d %H:%M:%S')
    orig_x =orig_x.set_index(['fecha'])
    print(F'Reading {file_name_y} ....')
    orig_y = pd.read_csv(join(input_folder,file_name_y))
    orig_y['fecha'] = pd.to_datetime(orig_y['fecha'],format='%Y-%m-%d %H:%M:%S')
    orig_y =orig_y.set_index(['fecha'])

    X = orig_x.values
    Y = orig_y.values

    curr_ids = 0 # First index to use
    while True:
        # The examples should already be shuffled
        if curr_ids < (len(example_ids) - batch_size):
            curr_ids += batch_size
        else:
            curr_ids = 0
            np.random.shuffle(example_ids) # We shuffle the folders every time we have tested all the examples

        curr_batch_x = X[example_ids[curr_ids:curr_ids+batch_size],:]
        curr_batch_y = Y[example_ids[curr_ids:curr_ids+batch_size],:]
        try:

            yield curr_batch_x, curr_batch_y

        except Exception as e:
            print("----- Not able to generate for: ", curr_ids, " ERROR: ", str(e))
