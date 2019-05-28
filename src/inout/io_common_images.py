import os
import SimpleITK as sitk

def copyItkImage(itk_src, np_arr):
    out_itk = sitk.GetImageFromArray(np_arr)
    out_itk.SetOrigin(itk_src.GetOrigin())
    out_itk.SetDirection(itk_src.GetDirection())
    out_itk.SetSpacing(itk_src.GetSpacing())
    return out_itk



