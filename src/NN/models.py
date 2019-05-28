from keras.layers import *
from keras.models import *

def getDefaultModel(n_in):
    '''
    :param n_in:
    :param n_hidden_layers: Array, number of hidden layers per layer
    :param neurons_in_hl:
    :param output_n:
    :param activations is an array of strings or functions indicating activation functions for hidden and output layers
    :return:
    '''

    input_layer = Input(shape=(n_in,),name=F'in_lay_{n_in}')

    a = Dense(n_in*2,name='in_lay',activation='sigmoid')(input_layer)
    b = Dense(n_in*2,name='hid_lay',activation='sigmoid')(a)
    c = Dense(1,name='out_lay',activation='sigmoid')(b)

    model = Model(inputs=input_layer, outputs=c)
    return model


def getDenseModel(n_in, n_hidden_layers, neurons_in_hl, output_n,activations):
    '''
    :param n_in:
    :param n_hidden_layers: Array, number of hidden layers per layer
    :param neurons_in_hl:
    :param output_n:
    :param activations is an array of strings or functions indicating activation functions for hidden and output layers
    :return:
    '''

    hidden_layers = []
    input_layer = Input(shape=(n_in,),name=F'in_lay_{n_in}')

    hidden_layers.append(Dense(neurons_in_hl[0],name='hid_lay_0',activation=activations[0])(input_layer))

    for c_hl in range(1,n_hidden_layers):
        print(F'Adding hidden layer {c_hl+1}..')
        # Add a hidden layer, with input the previous hidden layer
        hidden_layers.append(Dense(neurons_in_hl[c_hl],name=F'hid_lay_{c_hl}',activation=activations[0])(hidden_layers[c_hl-1]))

    output_layer = Dense(output_n,name=F'out_lay_{output_n}',activation=activations[1])(hidden_layers[-1]) # Add last hidden layer as input to output layer

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


