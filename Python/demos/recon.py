#!/usr/bin/env python
# coding: utf-8


'''
Windows: Open Anaconda prompt
conda create --name tigre_env -c anaconda -c ccpi -c conda-forge python tigre simpleitk ipykernel opencv astropy tomopy nibabel scikit-image scikit-learn scipy tqdm scikit-learn-intelex jupyter ipywidgets
conda activate tigre_env

conda list --export > conda-package-list.txt
conda create -n tigre_env --file conda-package-list.txt
'''

import shared_functions as s
import json
import math
import os
from multiprocessing import freeze_support

import nibabel as nib
import numpy as np
from astropy.convolution import Gaussian2DKernel
from tqdm import tqdm
import matplotlib.pyplot as plt


kernel = Gaussian2DKernel(x_stddev=2)


if __name__ == "__main__":
    freeze_support()  # needed for Windows - see https://stackoverflow.com/questions/63871662/python-multiprocessing-freeze-support-error

    drive = 'f:\\'
    # basefolder = os.path.join(drive,'jasper','data','20220726_scanseries')
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220812_BreastTissueFFPE')
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220825_LegoMan')
    base_folder = os.path.join(drive, 'jasper', 'data', '20220805_tumourWhateverBreast')

    base_json_file = os.path.join(base_folder, 'scan_settings.json')
    results_folder = os.path.join(base_folder, 'results_fillgap')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Make a list of globals for the reconstruction setting, and log them in a json file
    gReconParams = dict()
    gReconParams['pixels'] = 512  # (pixels)
    gReconParams['pixel_pitch'] = 0.055  # (mm)
    gReconParams['fill_gap'] = True
    gReconParams['median_filter'] = False
    gReconParams['bad_pixel_correction'] = True
    gReconParams['recon_voxels'] = (
        gReconParams['pixels'], gReconParams['pixels'], gReconParams['pixels'])  # number of voxels (vx)

    ''' TODO These should really be read from the JSON file! '''
    gReconParams['distance_source_detector'] = 188.347  # 9+9+30+100+30+9+1.347 (mm)
    gReconParams['z_stage_distance_mm'] = s.get_sample_z_from_first_scan_json(
        base_json_file)  # Varies between 0 and 100 mm
    gReconParams['distance_object_detector'] = 30 + \
        gReconParams['z_stage_distance_mm'] + 9+1.347  # (mm)
    gReconParams['detector_rotation'] = (math.radians(-0.3), 0., 0.)  # (mm)

    assert gReconParams['z_stage_distance_mm'] < 100 and gReconParams['z_stage_distance_mm'] >= 0

    DSD = gReconParams['distance_source_detector']
    DSO = DSD - gReconParams['distance_object_detector']

    # TODO have to explain the numbers ::: For 20220822_ffpe_whateverBreast
    a = 512 * 0.055 / (((DSD-DSO) / DSO) + 1)
    gReconParams['recon_size'] = (a, a, a)

    centre_of_rotation_offset_x_mm = 2.51
    centre_of_rotation_offset_y_mm = -0.24
    print(f'centre_of_rotation_offset_x_mm = {centre_of_rotation_offset_x_mm} (mm)')
    print(f'centre_of_rotation_offset_y_mm = {centre_of_rotation_offset_y_mm} (mm)')

    spectral_projs_th0, spectral_open_th0, spectral_projs_th1, spectral_open_th1, th0_list, th1_list, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets = \
        s.load_or_generate_data_arrays(base_json_file, base_folder, results_folder, gReconParams)

    # E.g. (9, 32, 512, 512) (Thresholds, # of open images, pixels, pixels)
    print(spectral_open_th0.shape)

    open_mean_th0 = np.mean(spectral_open_th0, axis=1)
    open_mean_th1 = np.mean(spectral_open_th1, axis=1)

    for i in range(open_mean_th0.shape[0]):
        # print(i, open_mean_th0.shape, open_mean_th0.shape[0])  # E.g. 0 (9, 512, 512) 9
        open_mean_th0[i, :, :] = open_mean_th0[i, :, :]/exp_time[i]
        open_mean_th1[i, :, :] = open_mean_th1[i, :, :]/exp_time[i]

    for i in range(spectral_projs_th0.shape[0]):
        spectral_projs_th0[i, :, :, :] = spectral_projs_th0[i, :, :, :] / exp_time[i]
        spectral_projs_th1[i, :, :, :] = spectral_projs_th1[i, :, :, :] / exp_time[i]

    '''
    Merging the open_means and th_lists for all thresholds
    '''
    open_mean_all_thresholds = list()
    combined_energy_list = list()
    # mean_of_thresholds = list()
    for idx in range(len(th0_list)):
        open_mean_all_thresholds.append(open_mean_th0[idx, :, :])
        open_mean_all_thresholds.append(open_mean_th1[idx, :, :])
        combined_energy_list.append(th0_list[idx])
        combined_energy_list.append(th1_list[idx])

    open_mean_all_thresholds = np.array(open_mean_all_thresholds)

    ofc_th0 = np.empty_like(spectral_projs_th0)
    ofc_th1 = np.empty_like(spectral_projs_th0)

    pixel_count_offsets = s.save_and_or_load_npy_files(
        results_folder, f'pixel_count_offsets.npy', lambda: s.generate_dac_values(gReconParams, open_mean_all_thresholds, combined_energy_list, plot=True))

    plt.imshow(pixel_count_offsets[0, :, :])
    plt.show()
    plt.imshow(spectral_projs_th0[0, 0, :, :])
    plt.show()
    plt.imshow(ofc_th0[0, 0, :, :])
    plt.show()

    # for idx in range(0, len(th0_list)):
    #     for p in range(ofc_th0.shape[0]):

    #         ofc_th1[idx, p, :, :] = - \
    #             np.log(spectral_projs_th1[idx, p, :, :] / open_mean_th1[idx, :, :])
    #         ofc_th0[idx, p, :, :] = - \
    #             np.log(spectral_projs_th0[idx, p, :, :] / open_mean_th0[idx, :, :])
    #         ofc_th1[idx, p, :, :] = - \
    #             np.log(spectral_projs_th1[idx, p, :, :] / open_mean_th1[idx, :, :])

    idx = 0
    for energy in range(0, len(combined_energy_list), 2):
        for p in range(ofc_th0.shape[0]):
            ofc_th0[idx, p, :, :] = spectral_projs_th0[idx,
                                                       p, :, :] - pixel_count_offsets[energy, :, :]
            ofc_th0[idx, p, :, :] = - \
                np.log(ofc_th0[idx, p, :, :] / open_mean_th0[idx, :, :])
            ofc_th1[idx, p, :, :] = spectral_projs_th1[idx,
                                                       p, :, :] - pixel_count_offsets[energy + 1, :, :]
            ofc_th1[idx, p, :, :] = - \
                np.log(ofc_th1[idx, p, :, :] / open_mean_th1[idx, :, :])

        plt.imshow(ofc_th0[0, 180, :, :])
        plt.show()

        print(f'Doing recon finally! Mean energy = {combined_energy_list[energy]} keV')
        img, geom = s.recon_scan(gReconParams, ofc_th0[idx, :, :, :], angles, detector_x_offsets,
                                 detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, True)
        ni_img = nib.Nifti1Image(img, np.eye(4))
        ni_img = s.make_Nifti1Image(img, geom.dVoxel)
        s.save_array(results_folder, 'Recon_'
                     + str(combined_energy_list[energy])+'OFC_NEWBOIII.nii', ni_img)
        idx += 1
