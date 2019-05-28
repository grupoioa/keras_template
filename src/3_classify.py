import preproc.utils as utils
import visualization.utilsviz as utilsviz
import models.models as models
from models.metrics import *
import re
import os
from pandas import DataFrame
from config.MainConfig import getMakeSegmentationConfig
from inout.io_mri import *


def makeSegmentation(inFolder, outputDirectory, outputImages, model_weights_file,
                        all_params, cases='all', save_segmentations=True):

    only_ROI = not(all_params['orig_resolution'])
    type_segmentation = all_params['type_segmentation']
    model_type = all_params['model_name']
    roi_slices = all_params['roi_slices']

    img_size = 168
    threshold = 0.5

    if model_type == "3dm":
        model = models.getModel_3D_Multi([img_size,img_size,img_size], 'sigmoid')
        multistream = True
    if model_type == "3dmorig":
        model = models.getModel_3D_MultiORIGINAL([img_size,img_size,img_size], 'sigmoid')
        multistream = True
    if model_type == "3ddropout":
        print("DROPUT*****************************")
        model = models.getModel_3D_Multi_Dropout([img_size,img_size,img_size], 'sigmoid')
        multistream = True

    model.load_weights(model_weights_file)

    # Define which cases are we going to perform the segmentation
    if isinstance(cases,str):
        if cases == 'all':
            examples = os.listdir(inFolder)
        else:
            examples = ['Case-{:04d}'.format(case) for case in cases]
    examples.sort()

    if not os.path.exists(outputImages):
        os.makedirs(outputImages)

    dsc_data = DataFrame(index = examples, columns=['ROI', 'Original'])

    for current_folder in examples:
        try:
            # Reads original image and prostate
            [img_tra_original, img_tra_HR, ctr_pro, ctr_pro_HR, roi_ctr_pro, startROI, sizeROI] = readImgAndProstate(inFolder, current_folder)
            # Reads PZ and input for NN
            if type_segmentation == 'PZ' or type_segmentation == 'Prostate':
                [ctr_pz, ctr_pz_HR, roi_ctr_pz] = readPZ(inFolder, current_folder, multistream, img_size)
            # Reads Lesion and input for NN
            if type_segmentation == 'Lesion':
                [ctr_lesion, ctr_lesion_HR, roi_ctr_lesion] = readLesion(inFolder, current_folder)

            [roi_img1, roi_img2, roi_img3] = readROI(inFolder, current_folder, type_segmentation)
            pro = sitk.GetArrayFromImage(roi_ctr_pro)

            # To visuali the ROI
            # utilsviz.drawMultipleSeriesItk([roi_img1, roi_img2,roi_img3 ], slices='middle')

            input_array = utils.createInputArray(multistream, img_size, roi_img1, roi_img2, roi_img3)

            # To visualithe final INPUT
            # utilsviz.drawMultipleSeriesNumpy([input_array[0,:,:,:,0], input_array[0,:,:,:,1],
            #                                   input_array[0,:,:,:,2]], slices=[20,21,22], colorbar=True,
            #                                  savefig='/media/osz1/DATA/DATA/GE/IMAGES/SEGMENTATION/PZ/DELETE/')

            # This is one of our parts, remove everything outside the prostate for PZ or lesion
            if type_segmentation == 'Lesion':
                # TODO check this part is multiplying by the prostate (input only inside prostate)
                input_array[0][0,:,:,:,0] = input_array[0][0,:,:,:,0] *pro
                input_array[1][0,:,:,:,0] = input_array[1][0,:,:,:,0] *pro
                input_array[2][0,:,:,:,0] = input_array[2][0,:,:,:,0] *pro

            # ************** NN Prediction ******************
            print('Predicting image {} ({})....'.format(current_folder, inFolder))
            if multistream:
                output_NN = model.predict(input_array, verbose=1)
                if type_segmentation == 'PZ' or type_segmentation == 'Lesion':
                    output_NN[0,:,:,:,0] = output_NN[0,:,:,:,0] * pro # Remove all outside prostate
            else:
                output_NN = model.predict(input_array, verbose=1)

            # Shows original output of the network
            # for i in np.arange(20, 140,4):
            #     plt.imshow(output_NN[0,i,:,:,0])
            #     plt.contour(output_NN[0,i,:,:,0])
            #     plt.show()

            pred_nn = sitk.GetImageFromArray(output_NN[0,:,:,:,0])
            pred_nn = utils.binaryThresholdImage(pred_nn, threshold)
            pred_nn = utils.getLargestConnectedComponents(pred_nn)

            c_img_folder = join(outputImages,current_folder)
            if type_segmentation == 'Prostate':
                cur_dsc_roi = numpy_dice(sitk.GetArrayFromImage(roi_ctr_pro), sitk.GetArrayFromImage(pred_nn))
            if type_segmentation == 'PZ':
                cur_dsc_roi = numpy_dice(sitk.GetArrayFromImage(roi_ctr_pz), sitk.GetArrayFromImage(pred_nn))

            print('--------------{} DSC ROI: {:02.2f} DSC Original {:02.2f} ------------'.format(c_img_folder, cur_dsc_roi, cur_dsc_roi))

            slices = roi_slices
            title = '{} {} DSC {:02.2f}'.format(type_segmentation, current_folder, cur_dsc_roi)
            if type_segmentation == 'Lesion':
                utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, title=title, contours=[roi_ctr_pro, roi_ctr_lesion, pred_nn],
                                   savefig=join(outputImages,'ROI_LESION_'+current_folder), labels=['Prostate','GT','NN'])
            if type_segmentation == 'Prostate':
                utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, title=title, contours=[roi_ctr_pro, pred_nn],
                                    savefig=join(outputImages,'ROI_PROSTATE_'+current_folder), labels=['GT','NN'])
            if type_segmentation == 'PZ':
                utilsviz.drawMultipleSeriesItk([roi_img1], slices=slices, title=title, contours=[roi_ctr_pro, roi_ctr_pz, pred_nn],
                                     savefig=join(outputImages,'ROI_PZ_'+current_folder), labels=['Prostate','GT','NN'])

            if save_segmentations:
                print('Saving original prediction (ROI)...')
                if not os.path.exists(join(outputDirectory, current_folder)):
                    os.makedirs(join(outputDirectory, current_folder))
                sitk.WriteImage(pred_nn, join(outputDirectory, current_folder, 'predicted_roi.nrrd'))

            dsc_data.loc[current_folder]['ROI'] = cur_dsc_roi

             # Ploting DSC for ROI
            dsc_data.loc['AVG'] = dsc_data.mean()
            title = 'DSC AVG ROI: {:.3f} '.format(dsc_data.loc['AVG']['ROI'])
            utilsviz.plotMultipleDSC([dsc_data['ROI'].dropna().to_dict()],
                                     title=title, legends=['ROI'],
                                     savefig=join(outputImages,'aroi_DSC.png'))
            dsc_data.to_csv(join(outputImages,'aroi_DSC.csv'))

            if not(only_ROI): # in this case we do not upscale, just show the prediction
                output_predicted_original = sitk.Image(img_tra_HR.GetSize(), sitk.sitkFloat32)
                arr = sitk.GetArrayFromImage(output_predicted_original) # Gets an array same size as original image
                arr[:] = 0 # Make everything = 0
                arr[startROI[2]:startROI[2]+sizeROI[2], startROI[1]:startROI[1]+sizeROI[1],startROI[0]:startROI[0]+sizeROI[0]] = output_NN[0,:,:,:,0]
                output_predicted = sitk.GetImageFromArray(arr)
                output_predicted = utils.binaryThresholdImage(output_predicted, threshold)
                output_predicted = utils.getLargestConnectedComponents(output_predicted)
                output_predicted = sitk.BinaryFillhole(output_predicted, fullyConnected=True)
                output_predicted.SetOrigin(img_tra_HR.GetOrigin())
                output_predicted.SetDirection(img_tra_HR.GetDirection())
                output_predicted.SetSpacing(img_tra_HR.GetSpacing())
                if save_segmentations:
                    sitk.WriteImage(output_predicted, join(outputDirectory, current_folder, 'predicted_HR.nrrd'))

                # original transversal space (high slice thickness), transform perdiction with shape-based interpolation (via distance transformation)
                # segm_dis = sitk.SignedMaurerDistanceMap(output_predicted, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
                # segm_dis = utils.resampleToReference(output_predicted, img_tra_original, sitk.sitkLinear, -1) # TODO don't know why it had -1 here
                segm_dis = utils.resampleToReference(output_predicted, img_tra_original, sitk.sitkNearestNeighbor, 0)
                thresholded = utils.binaryThresholdImage(segm_dis, threshold)
                thresholded = utils.getLargestConnectedComponents(thresholded)
                if save_segmentations:
                    sitk.WriteImage(thresholded, join(outputDirectory, current_folder, 'predicted_transversal_space.nrrd'))

                if type_segmentation == 'Prostate':
                    cur_dsc_original = numpy_dice(sitk.GetArrayFromImage(ctr_pro), sitk.GetArrayFromImage(thresholded))
                if type_segmentation == 'PZ':
                    cur_dsc_original = numpy_dice(sitk.GetArrayFromImage(ctr_pz), sitk.GetArrayFromImage(thresholded))

                title = '{} DSC {:02.3f}'.format(current_folder, cur_dsc_original)
                slices = all_params['orig_slices']
                if type_segmentation == 'Lesion':
                    utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, ctr_pz, thresholded],
                                           labels=['Prostate','GT','NN'], savefig=join(outputImages, current_folder))

                if type_segmentation == 'PZ':
                    utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, ctr_pz, thresholded],
                                           labels=['Prostate','PZ','NN'], savefig=join(outputImages, current_folder))

                if type_segmentation == 'Prostate':
                    utilsviz.drawSeriesItk(img_tra_original, slices=slices, title=title, contours=[ctr_pro, thresholded],
                                           labels=['GT','NN'], savefig=join(outputImages, current_folder))

                # Ploting DSC for ROI
                dsc_data.loc[current_folder]['Original'] = cur_dsc_original
                dsc_data.loc['AVG'] = dsc_data.mean()
                title = 'DSC AVG ROI: {:.3f} Original: {:.3f}'.format(dsc_data.loc['AVG']['ROI'],dsc_data.loc['AVG']['Original'])
                utilsviz.plotMultipleDSC([dsc_data['ROI'].dropna().to_dict(), dsc_data['Original'].dropna().to_dict()],
                                         title=title, legends=['ROI', 'Original'],
                                         savefig=join(outputImages,'all_DSC.png'))

                dsc_data.to_csv(join(outputImages,'all_DSC.csv'))

        except Exception as e:
            print("---------------------------- Failed {} error: {} ----------------".format(current_folder, e))

if __name__ == '__main__':

    all_params = getMakeSegmentationConfig()

    run_mode= all_params['mode']
    only_ROI = not(all_params['orig_resolution'])
    type_segmentation =  all_params['type_segmentation']
    model_name = all_params['model_name']
    disp_images = all_params['display_images']
    cases = all_params['cases']
    save_segmentations = all_params['save_segmentations']
    utilsviz.view_results = disp_images

    if run_mode == 'single':
        outputImages = all_params['output_images']
        inputDirectory = all_params['input_folder']
        outputDirectory = all_params['output_folder']
        model_weights_file = all_params['weights']
        makeSegmentation(inputDirectory, outputDirectory,
                         outputImages, model_weights_file,
                         all_params, cases=cases, save_segmentations=save_segmentations)

    if run_mode == 'multiple':
        # ************************ This part is to make the segmenation of ALL THE TESTS ****************
        input_root_folder= all_params['input_root_folder']
        output_root_data = all_params['output_root_data']
        output_root_imgs = all_params['output_root_imgs']
        modelWeightsFolder = all_params['weights_folder']

        datasets = all_params['input_folder_names']
        outputDirectoriesByModel = all_params['weight_file_names']

        # Add PZ or Lesion into the model folders
        if type_segmentation == 'PZ'or type_segmentation == 'Lesion':
            outputDirectoriesByModel = ['{}_{}'.format(x,type_segmentation) for x in outputDirectoriesByModel]

        modelWeightsNames = [x+'\S*.hdf5' for x in outputDirectoriesByModel]

        all_model_files = os.listdir(modelWeightsFolder)
        # Iterate over all the model files we want to use
        for exIdx, cur_weights in enumerate(modelWeightsNames):
            # Here we are searching the model file by regular expression
            matched_folders = [x for x in all_model_files if not (re.search(cur_weights, x) is None)]
            if len(matched_folders) > 0:
                print('Matched model file:',matched_folders[0])
                model_weights_file = join(modelWeightsFolder,matched_folders[0]) # Get the final model file to use
            else:
                print('No matched model file found for {}'.format(cur_weights))
                continue
            # Iterate over the datasets to use
            for magIdx, cur_dataset in enumerate(datasets):
                # Make final outputdirectory folder name
                inputDirectory = join(input_root_folder,cur_dataset, 'Preproc')
                complete_outputDirectory = join(output_root_data, outputDirectoriesByModel[exIdx])
                final_outputImages = join(output_root_imgs, cur_dataset,type_segmentation, outputDirectoriesByModel[exIdx])
                print("*********************************************** {} {}".format(exIdx, magIdx))
                print('Weights {}'.format(model_weights_file))
                print("InputDir: {}".format(inputDirectory))
                print("Outdir: {}".format(complete_outputDirectory))
                print("Outimgs: {}".format(final_outputImages))

                makeSegmentation(inputDirectory, complete_outputDirectory, final_outputImages,
                                 model_weights_file, all_params, cases=cases[magIdx], save_segmentations=save_segmentations)
