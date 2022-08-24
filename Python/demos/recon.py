#!/usr/bin/env python
# coding: utf-8


'''
Windows: Open Anaconda prompt
conda create --name tigre_env -c anaconda -c ccpi -c conda-forge python tigre simpleitk ipykernel opencv astropy tomopy nibabel scikit-image scikit-learn scipy tqdm scikit-learn-intelex jupyter ipywidgets
conda activate tigre_env

conda list --export > conda-package-list.txt
conda create -n tigre_env --file conda-package-list.txt
'''

import json
import math
import os
from multiprocessing import freeze_support

import nibabel as nib
import numpy as np
from astropy.convolution import Gaussian2DKernel
from tqdm import tqdm

import shared_functions as s

kernel = Gaussian2DKernel(x_stddev=2)


if __name__ == "__main__":
    freeze_support()  # needed for Windows - see https://stackoverflow.com/questions/63871662/python-multiprocessing-freeze-support-error

    drive = 'f:\\'
    # basefolder = os.path.join(drive,'jasper','data','20220726_scanseries')
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220812_BreastTissueFFPE')
    base_folder = os.path.join(drive, 'jasper', 'data', '20220822_Al_Phantom_Recon_Alignment')
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
    gReconParams['recon_voxels'] = (gReconParams['pixels'], gReconParams['pixels'],
                                    gReconParams['pixels'])  # number of voxels (vx)
    gReconParams['recon_size'] = (gReconParams['pixels'] * gReconParams['pixel_pitch'], gReconParams['pixels']
                                  * gReconParams['pixel_pitch'], gReconParams['pixels'] * gReconParams['pixel_pitch'])  # (mm)

    gReconParams['distance_source_detector'] = 188.347  # 9+9+30+100+30+9+1.347 (mm)
    gReconParams['z_stage_distance_mm'] = 20  # Varies between 0 and 100 mm
    gReconParams['z_stage_distance_mm'] = s.get_sample_z_from_first_scan_json(
        base_json_file)  # Varies between 0 and 100 mm
    gReconParams['distance_object_detector'] = 30 + \
        gReconParams['z_stage_distance_mm'] + 9+1.347  # (mm)
    gReconParams['detector_rotation'] = (math.radians(-0.3), 0., 0.)  # (mm)

    assert gReconParams['z_stage_distance_mm'] < 100 and gReconParams['z_stage_distance_mm'] > 0

    ''' Load tiff files, transform a bit and save to numpy files. Load the files if they exist already '''
    if os.path.exists(base_json_file):
        f = open(base_json_file)
        dashboard = json.load(f)
        exp_time = []
        th0_list = []
        th1_list = []

        numpy_output_files = ['projs_stack_th0.npy', 'open_stack_th0.npy',
                              'projs_stack_th1.npy', 'open_stack_th1.npy', 'thlist_th0.npy', 'thlist_th1.npy']
        if s.files_exist(results_folder, numpy_output_files):
            print('Loading existing numpy files, should take ~7.5 seconds')

            spectral_projs_th0 = np.load(os.path.join(results_folder, numpy_output_files[0]))
            spectral_open_th0 = np.load(os.path.join(results_folder, numpy_output_files[1]))
            spectral_projs_th1 = np.load(os.path.join(results_folder, numpy_output_files[2]))
            spectral_open_th1 = np.load(os.path.join(results_folder, numpy_output_files[3]))
            th0_list = np.load(os.path.join(results_folder, numpy_output_files[4]))
            th1_list = np.load(os.path.join(results_folder, numpy_output_files[5]))

            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                open_image_folder = scan_folder
                open_image_json = scan_json
                folder_string = dashboard['thresholdscan'][i]['projectionsfolder']

                th0_keV = folder_string[0:folder_string.find('_')]
                th1_keV = folder_string[folder_string.find('_')+1:]

                angles = s.get_proj_angles(scan_json)
                z_offset = s.get_samplestage_z_offset(scan_json)
                detector_x_offsets, detector_y_offsets = s.get_detector_offsets(scan_json)
                exp_time.append(s.get_exposure_time_projection(scan_json))
            exp_time = np.asarray(exp_time)

        else:
            print(f'Making new numpy files, should take ~4.5 minutes. At least one file was missing :( ')

            spectral_projs_th0 = []
            spectral_open_th0 = []
            spectral_projs_th1 = []
            spectral_open_th1 = []
            th0_list = []
            th1_list = []
            th1_exp_time = []
            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                open_image_folder = scan_folder
                open_image_json = scan_json
                folder_string = dashboard['thresholdscan'][i]['projectionsfolder']
                th0_keV = folder_string[0:folder_string.find('_')]
                th1_keV = folder_string[folder_string.find('_')+1:]
                exp_time.append(s.get_exposure_time_projection(scan_json))
                th0_list.append(float(th0_keV))
                th1_list.append(float(th1_keV))

                projs_th0 = s.projectionsloader(
                    scan_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                openimg_th0 = s.openimgloader(
                    open_image_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                projs_th1 = s.projectionsloader(
                    scan_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                openimg_th1 = s.openimgloader(
                    open_image_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                spectral_projs_th0.append(projs_th0)
                spectral_open_th0.append(openimg_th0)
                spectral_projs_th1.append(projs_th1)
                spectral_open_th1.append(openimg_th1)
                detector_x_offsets, detector_y_offsets = s.get_detector_offsets(scan_json)
                angles = s.get_proj_angles(scan_json)
                z_offset = s.get_samplestage_z_offset(scan_json)
            spectral_projs_th0 = np.asarray(spectral_projs_th0)
            spectral_open_th0 = np.asarray(spectral_open_th0)
            spectral_projs_th1 = np.asarray(spectral_projs_th1)
            spectral_open_th1 = np.asarray(spectral_open_th1)
            exp_time = np.asarray(exp_time)
            th0_list = np.asarray(th0_list)
            th1_list = np.asarray(th1_list)

            np.save(os.path.join(results_folder, 'projs_stack_th0.npy'), spectral_projs_th0)
            np.save(os.path.join(results_folder, 'open_stack_th0.npy'), spectral_open_th0)
            np.save(os.path.join(results_folder, 'projs_stack_th1.npy'), spectral_projs_th1)
            np.save(os.path.join(results_folder, 'open_stack_th1.npy'), spectral_open_th1)
            np.save(os.path.join(results_folder, 'thlist_th0.npy'), th0_list)
            np.save(os.path.join(results_folder, 'thlist_th1.npy'), th1_list)

            # print(spectralprojs.shape, spectralopen.shape)
            # out = np.load(os.path.join(basefolder,'Projections_fitted.npy'))
            # openout = np.load(os.path.join(basefolder,'Projections_open_fitted.npy'))

    # print(spectral_projs_th0.shape)

    # _global_detector_shift_y = find_optimal_offset(gReconParams, spectral_projs_th0[1, :, :, :], angles, detector_x_offsets, detector_y_offsets, stage_offset=0, search_range=20)
    # print(_global_detector_shift_y)
    # global_detector_shift_y=0.12

    # print(f'global_detector_shift_x = {global_detector_shift_x}')
    # print(f'global_detector_shift_y = {global_detector_shift_y}')
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

    # global_detector_shift_y = -0.31872500000000004

    for th in range(0, open_mean_th0.shape[0]):
        a0 = 0
        a1 = (gReconParams['pixels'] // 2) - 1  # 255 for 512 pixels
        a2 = (gReconParams['pixels'] // 2) + 1  # 257 for 512 pixels
        a3 = gReconParams['pixels']
        mean = 0.25 * (
            np.mean(open_mean_th0[th, a0:a1, a0:a1])
            + np.mean(open_mean_th0[th, a0:a1, a2:a3])
            + np.mean(open_mean_th0[th, a2:a3, a0:a1])
            + np.mean(open_mean_th0[th, a2:a3, a2:a3])
        )
        print(f'th = {th}, mean = {mean}')  # E.g. 36895.64768079413

        print('Finding best DAC values per pixel...')
        DAC_values = s.save_and_or_load_npy_files(
            results_folder, f'th{th}_dac_values.npy', lambda: s.generate_dac_values(gReconParams, open_mean_th0, th0_list, mean, th))

        fit_array = spectral_projs_th0.reshape(spectral_projs_th0.shape[0], -1)

        print('Fitting polynomials... ~1 minute')
        regressions, _, _, _, _ = np.polyfit(th0_list, fit_array, 2, full=True)
        # regressions = save_and_or_load_npy_files(results_folder, f'th{th}_regressions.npy', lambda: FILLMEIN(...))

        print('Calculating DACs...')
        calc_dacs = np.repeat(DAC_values[np.newaxis, :, :],
                              spectral_projs_th0.shape[1], axis=0).flatten()

        print('Calculating projection data...')
        proj_data_flat = (regressions[0, :]*calc_dacs**2) + \
            (regressions[1, :]*calc_dacs**1) + regressions[2, :]
        projection_data = proj_data_flat.reshape(
            spectral_projs_th0.shape[1], spectral_projs_th0.shape[2], spectral_projs_th0.shape[3])
        s.save_array(results_folder, 'Projections_th0_'
                     + str(th0_list[th])+'_keV.npy', projection_data)

        # TODO Ask Jasper about this VERY SUSPICIOUS CODE
        # mean = (np.mean(projection_data[:, :, 0:10]) + np.mean(projection_data[:, :, 503:513]))/2
        ofc = -np.log(projection_data/1)
        s.save_array(results_folder, 'projs_th0_'+str(th0_list[th])+'OFC_interp.npy', ofc)

        ofc_bpc = s.save_and_or_load_npy_files(
            results_folder, f'th{th}_bpc.npy', lambda: s.generate_bad_pixel_corrected_array(ofc, gReconParams))

        ofc_bpc_mf = s.median_filter_projection_set(ofc_bpc, 3)  # TODO THIS IS FUCKED
        if th == 0:
            # print(f'th = {th}, finding optimal offset')
            # (mm) # find_optimal_offset(gReconParams, spectral_projs_th0[1, :, :, :], angles, detector_x_offsets, detector_y_offsets, stage_offset=0, search_range=25)
            centre_of_rotation_offset_x_mm = 2.51
            centre_of_rotation_offset_y_mm = 0
            print(f'centre_of_rotation_offset_x_mm = {centre_of_rotation_offset_x_mm} (mm)')
            print(f'centre_of_rotation_offset_y_mm = {centre_of_rotation_offset_y_mm} (mm)')

            ni_img = nib.Nifti1Image(ofc_bpc_mf, np.eye(4))
            s.save_array(results_folder, 'Proj_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)

        print('Doing recon finally!')
        img_th0 = s.recon_scan(gReconParams, ofc_bpc, angles, z_offset, detector_x_offsets,
                               detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm)

        ni_img = nib.Nifti1Image(img_th0, np.eye(4))
        s.save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC.nii', ni_img)

        img_th0 = s.recon_scan(gReconParams, ofc_bpc_mf, angles, z_offset, detector_x_offsets,
                               detector_y_offsets, centre_of_rotation_offset_y_mm)
        ni_img = nib.Nifti1Image(img_th0, np.eye(4))
        s.save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)
