import keras.callbacks as callbacks
from os.path import join

def getAllCallbacks(model_name, early_stopping, weights_folder='.', logs_folder='.', min_delta=.001, patience=70):
    root_logdir = join(logs_folder,'logs')
    logdir = "{}/run-{}/".format(root_logdir, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.2f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model,monitor=early_stopping, save_best_only=True,
                                              mode='max',save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping, min_delta=min_delta, patience=patience, mode='max')
    return [logger, save_callback, stop_callback]

# def runAndSaveModel(model, X, Y, model_name, epochs, is3d, val_per, early_stopping) :
#
#     # plot_model(model, to_file='../images/{}.png'.format(model_name), show_shapes=True)
#
#     [logger, save_callback, stop_callback] = getAllCallbacks(model_name, early_stopping)
#
#     if is3d:
#         model.fit(
#             x = X,
#             y = Y,
#             epochs=epochs,
#             shuffle=True,
#             batch_size=1,
#             verbose=2,
#             validation_split=val_per,
#             callbacks=[logger, save_callback, stop_callback]
#         )
#     else:
#         model.fit(
#             x = X,
#             y = Y,
#             epochs=epochs,
#             shuffle=True,
#             batch_size=4,
#             verbose=2,
#             validation_split=val_per,
#             callbacks=[logger]
#         )

# def runModelWithGenerator(model, path, foders_to_read, model_name, epochs, val_per, early_stopping):
#
#     root_logdir = "../logs"
#     logdir = "{}/run-{}/".format(root_logdir, model_name)
#
#     logger = callbacks.TensorBoard(
#         log_dir=logdir,
#         write_graph=True,
#         write_images=True,
#         histogram_freq=0
#     )
#
#     # Saving the model every epoch
#     filepath_model = '../models/'+model_name+'---{epoch:02d}-{val_loss:.2f}.hdf5'
#     save_callback = callbacks.ModelCheckpoint(filepath_model,monitor=early_stopping, save_best_only=True,
#                                               mode='max',save_weights_only=True)
#
#     # Early stopping
#     stop_callback = callbacks.EarlyStopping(monitor=early_stopping, min_delta=.001, patience=150, mode='max')
#
#     model.fit_generator( data_gen_multi(path, folders_to_read, scan_names, roi_names)
#                         y = Y,
#                         epochs=epochs,
#                         shuffle=True,
#                         batch_size=1,
#                         callbacks=[logger, save_callback, stop_callback]
#     )

def trainMultipleModels(dataContainer, imgs_dims, model_name, architectures=['2D','simple', '2DU', '2DSec']):

    optimizers_str = ['sgd']
    #losses = [dice_coef_loss, 'mean_squared_error', 'categorical_crossentropy']
    losses = [dice_coef_loss]
    metrics = [real_dice_coef, 'accuracy']
    epochs = 1000

    # For SGD the hyperams are 1) learning rate 2) decay
    #r = -4*np.random.random(10) # from .0001  to 1 (log)
    #rd = -4*np.random.random(10)# from .0001  to 1 (log)

    #hyperpar = {'sgd': {'lr': 10**r, 'decay': 10**rd}}
    # hyperpar = {'sgd': {'lr': 6*np.random.random(3), 'decay': .01*np.random.random(3), \
    #                     'mom':  .8+.2*np.random.random(3)}}

    hyperpar = {'sgd': {'lr': [.001], 'decay': [.0001], 'mom': [.9]}}

    for optim in optimizers_str:
        if optim == 'sgd':
            for dc in hyperpar.get(optim).get('decay'):
                for mom in hyperpar.get(optim).get('mom'):
                    for lr in hyperpar.get(optim).get('lr'):
                        sgd = SGD(lr=lr, decay=dc, momentum=mom)

                        for loss in losses:
                            if callable(loss):
                                loss_str = loss.__name__
                            else:
                                loss_str = loss

                            for curr_arc in architectures:
                                is3d = False
                                if curr_arc == 'simple':
                                    model = simpleModel(imgs_dims)

                                if curr_arc == '2D':
                                    model = trainModel2D(imgs_dims)

                                if curr_arc == '2DSec':
                                    model = trainModel2DSec(imgs_dims)

                                if curr_arc == '3D_single':
                                    model = getModel_3D_Single(imgs_dims)
                                    is3d = True

                                if curr_arc == '2DU':
                                    model = trainModel2DUNet(imgs_dims)

                                f_mod_name = "{}_Opt_{}_lr_{:2.3f}_mom_{:2.3f}_decay_{:2.4f}_{}_{}".format(\
                                                curr_arc, optim, lr, mom, dc, loss_str, model_name)
                                print("\n Compiling ******** {} ******".format(f_mod_name))
                                model.compile(loss=loss, optimizer=optim, metrics=metrics)

                                print("Fitting .....")
                                runAndSave(model, dataContainer, f_mod_name, epochs, is3d)

    return model

def saveSplits(file_name, folders_to_read, train_idx, val_idx, test_idx):
    print("Saving splits....")
    file = open(file_name,'w')
    file.write("\n\nTrain examples (total:{}) :{}".format(len(train_idx), folders_to_read[train_idx]))
    file.write("\n\nValidation examples (total:{}) :{}:".format(len(val_idx), folders_to_read[val_idx]))
    file.write("\n\nTest examples (total:{}) :{}".format(len(test_idx), folders_to_read[test_idx]))
    file.close()
