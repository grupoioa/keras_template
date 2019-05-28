import numpy as np
import preproc.utils as utils
from scipy.ndimage.filters import gaussian_filter

def flipping(imgs, ctrs):
    faxis = 3
    imgs[:,0,:,:,:] = np.flip(imgs[:,0,:,:,:], axis = faxis)
    imgs[:,1,:,:,:] = np.flip(imgs[:,1,:,:,:], axis = faxis)
    imgs[:,2,:,:,:] = np.flip(imgs[:,2,:,:,:], axis = faxis)
    ctrs[:,0,:,:,:] = np.flip(ctrs[:,0,:,:,:], axis = faxis)
    return imgs, ctrs

def gaussblur(imgs):
    sigma = 2*np.random.random() # Blur from 0 to 2
    # print("Gaussian blur, sigma: {}".format(sigma))
    for i in range(len(imgs)):
        imgs[i,0,:,:,:] = gaussian_filter(imgs[i,0,:,:,:], sigma=sigma, order=0, mode='mirror')
        imgs[i,1,:,:,:] = gaussian_filter(imgs[i,1,:,:,:], sigma=sigma, order=0, mode='mirror')
        imgs[i,2,:,:,:] = gaussian_filter(imgs[i,2,:,:,:], sigma=sigma, order=0, mode='mirror')

    return imgs

def shifting(imgs, ctrs):
    maxShift = .10 # Maximum shifting allowed in percentage
    shiftSize = int(imgs.shape[3]*maxShift*(np.random.random()*2 - 1))
    shiftAxis = np.random.randint(0,3)
    # print("Shifting {} Axis {}".format(shiftSize, shiftAxis))
    if shiftSize != 0:
        for i in range(len(imgs)):
            imgs[i,0,:,:,:] = utils.shift3D(imgs[i,0,:,:,:], shiftSize, shiftAxis)
            imgs[i,1,:,:,:] = utils.shift3D(imgs[i,1,:,:,:], shiftSize, shiftAxis)
            imgs[i,2,:,:,:] = utils.shift3D(imgs[i,2,:,:,:], shiftSize, shiftAxis)

        for i in range(len(ctrs)):
            ctrs[i,0,:,:,:] = utils.shift3D(ctrs[i,0,:,:,:], shiftSize, shiftAxis)

    return imgs, ctrs