from keras.layers import *
from keras.models import *

def two_cnn_3d(input, num_filters, fsize, make_pool=True):
    conv = Conv3D(num_filters, (fsize, fsize,fsize), padding='same', activation='relu')(input)
    conv = BatchNormalization(axis=4)(conv)
    conv = Conv3D(num_filters, (fsize, fsize,fsize), padding='same', activation='relu')(input)
    conv = BatchNormalization(axis=4)(conv)
    if make_pool:
        pool = MaxPooling2D(pool_size=(2, 2, 2))(conv)
    else:
        pool = []
    return [conv, pool]

def two_cnn(input, num_filters, fsize, make_pool=True):
    conv = Conv2D(num_filters, (fsize, fsize), padding='same', activation='relu')(input)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(num_filters, (fsize, fsize), padding='same', activation='relu')(conv)
    conv = BatchNormalization(axis=3)(conv)
    if make_pool:
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        pool = []
    return [conv, pool]


def make_up(input, match_lay, num_filters, fsize):
    '''
    :param input: Is the CNN we will upsample
    :param match_lay: Is the CNN layer in the other side of the U (we will concatenate)
    :param num_filters:
    :param fsize:
    :return:
    '''
    up = concatenate(
        [
            Conv2D(num_filters, (fsize,fsize), activation='relu', \
                   padding='same')(UpSampling2D(size=(2, 2))(input)),
            match_lay
        ], axis=3)
    [conv, dele] = two_cnn(up, int(num_filters), fsize, make_pool=False)
    return conv

def makeU(inputs, num_filters, fsize, num_levels):
    convs = []
    pools = []
    dele = []

    # Going down
    for level in range(num_levels+1):
        if level == 0:
            c_input = inputs
        else:
            c_input = pools[-1]

        if level == num_levels:
            m_pool = False
        else:
            m_pool = True

        [convT, poolT] = two_cnn(c_input, num_filters*(2**level), fsize, make_pool=m_pool)
        convs.append(convT)
        pools.append(poolT)

    for level in range(num_levels):
        convT = make_up(convs[-1], convs[num_levels-level-1], num_filters*(2**(num_levels-level-1)), fsize)
        convs.append(convT)

    return convs[-1]

