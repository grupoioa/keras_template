import NN.models as models
from datetime import datetime
import pandas as pd
from NN.ozcallbacks import *
from keras.optimizers import *
import NN.utilsNN as utilsNN
from config.MainConfig import getTrainingConfig
from os.path import join
from keras.metrics import *
from inout.dataGenerator import *


if __name__ == '__main__':

    training_options = getTrainingConfig()

    input_folder = training_options['input_folder']
    output_folder = training_options['output_folder']

    # TODO. Because we are using generators, I'm not sure how can I read this information automatically
    # tot_examples = 24708
    tot_examples = 82
    tot_cols = 917

    train_ids = []
    val_ids = []
    test_ids = []
    val_perc = .1
    test_perc = 0
    # ================ Split definition
    [train_ids, val_ids, test_ids] = utilsNN.splitTrainValidationAndTest(tot_examples, val_perc=val_perc, test_perc=test_perc)

    # ================ Reads the model
    print("Setting the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    n_in = tot_cols
    n_hidden_layers = 2
    neurons_in_hl = [tot_cols]*n_hidden_layers
    output_n = 1
    activations = ['sigmoid','sigmoid'] # Activations for hidden and output layers

    # model = models.getDenseModel(n_in, n_hidden_layers, neurons_in_hl, output_n, activations)
    model = models.getDefaultModel(n_in)
    model_name = F'sopas_{now}'

    # ================ Configures the optimization
    print("Configuring optimization....")
    loss="mean_squared_error"

    optimizer = SGD(lr=0.03, momentum=0.3, decay=0.0, nesterov=False)
    # optimizer='sgd' #'adam', 'sgd'

    # eval_metrics=[loss]
    model.compile(loss=loss, optimizer=optimizer)

    # all_callbacks = getAllCallbacks(model_name, 'val_{}'.format(eval_metrics[0].__name__), join(output_folder,'weights'),
    [logger, save_callback, stop_callback] = getAllCallbacks(model_name, loss,
                                                             join(output_folder,'weights'), join(output_folder))

    # ================ Trains the model
    generator = True
    batch_size = 1000
    epochs = 1000
    min_val_steps = 100

    if generator:
        options={
            'input_folder': input_folder,
        }

        train_generator = dataGenerator(options, train_ids, batch_size=batch_size)
        val_generator = dataGenerator(options, val_ids, batch_size=batch_size)

        model.fit_generator(train_generator, steps_per_epoch=min(30,len(train_ids)),
                        validation_data= val_generator,
                        validation_steps=min(min_val_steps,len(val_ids)),
                        # use_multiprocessing=False,
                        # workers=1,
                        use_multiprocessing=True,
                        workers=2,
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])

    else:

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

        print("Training the model....")
        model.fit(
            x=X[train_ids],
            y=Y[train_ids],
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            verbose=2,
            validation_split=val_perc,
            callbacks=[logger, save_callback, stop_callback]
        )

