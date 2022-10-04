#!/usr/bin/env python
# coding: utf-8

'''
Windows: Open Anaconda prompt
python -m pip install PyQt5

conda create --name tigre_env -c anaconda -c ccpi -c conda-forge  python tigre simpleitk ipykernel opencv astropy tomopy nibabel scikit-image scikit-learn scipy tqdm scikit-learn-intelex jupyter ipywidgets imageio
conda update conda
conda activate tigre_env

Should be done once only to set the environment variable
setx CONDA_DLL_SEARCH_MODIFICATION_ENABLE 1

conda list --export > conda-package-list.txt
conda create -n tigre_env --file conda-package-list.txt

conda env remove -n tigre_env
'''

import math
import os

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import trange

import shared_functions as s

if __name__ == "__main__":
    '''
    1. Load 1 dataset of projections + open field images
    2. Scan over a parameter with one reconstruction each

    TODO Optional argument for using detector motion

    '''
    scan_over_a_paramter = True
    th = 0

    # Make folders and load files for recon
    drive = 'f:\\'
    base_folder = os.path.join(drive, 'jasper', 'data', '20221003_Legoman_al_filter_100_um')

    # 20220822_ffpe_WhateverBreast        .... FILL ME IN ....
    # 20220822_Al_Phantom_Recon_Alignment det_rot=(0 to -0.5, 0, 0) x= 5.02 y=-0.24
    # 20220805_tumourWhateverBreast       det_rot=(0 to -0.5, 0, 0) x=-0.27 y=-0.24

    base_json_file = os.path.join(base_folder, 'scan_settings.json')
    output_folder = os.path.join(base_folder, 'parameter_scanning')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gReconParams = dict()

    gReconParams['pixels'] = 512  # (pixels)
    gReconParams['pixel_pitch'] = 0.055  # (mm)
    gReconParams['fill_gap'] = True
    gReconParams['median_filter'] = False
    gReconParams['bad_pixel_correction'] = True
    gReconParams['recon_voxels'] = (
        gReconParams['pixels'], gReconParams['pixels'], gReconParams['pixels'])  # number of voxels (vx)
    gReconParams['distance_source_detector'] = 188.090
    gReconParams['z_stage_distance_mm'] = s.get_sample_z_from_first_scan_json(base_json_file) # Varies between 0 and 100 mm
    gReconParams['distance_object_detector'] = 30 + \
        gReconParams['z_stage_distance_mm'] + 9+0.055+1.035  # (mm)
    gReconParams['detector_rotation'] = (math.radians(0.), math.radians(0.), math.radians(0.))  # (mm)

    assert gReconParams['z_stage_distance_mm'] < 100 and gReconParams['z_stage_distance_mm'] >= 0


    DSD = gReconParams['distance_source_detector']
    DSO = DSD - gReconParams['distance_object_detector']
    a = gReconParams['pixels'] * gReconParams['pixel_pitch'] / (((DSD-DSO) / DSO) + 1)
    gReconParams['recon_size'] = (a, a, a)
    print(gReconParams)

    centre_of_rotation_offset_x_mm = 0.1155
    centre_of_rotation_offset_y_mm = 0.  # Could be 0, not sure yet

    ''' Load tiff files, transform a bit and save to numpy files.
    Load the files if they exist already '''
    spectral_projs_th0, spectral_open_th0, _, _, _, _, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets, _, _ = s.load_or_generate_data_arrays(
        base_json_file, base_folder, output_folder, gReconParams)

    if 'Legoman_al_filter_100_um_' in base_folder:
        spectral_projs_th0 = np.flip(np.flip(spectral_projs_th0, axis=2), axis=3)
        spectral_open_th0  = np.flip(np.flip(spectral_open_th0, axis=2), axis=3)

    open_mean_th0 = np.mean(spectral_open_th0, axis=1)
    open_mean_th0[0, :, :] = open_mean_th0[0, :, :]/exp_time[0]
    spectral_projs_th0[0, :, :, :] = spectral_projs_th0[0, :, :, :] / exp_time[0]
    ofc = -np.log(spectral_projs_th0[0, :, :, :] / open_mean_th0[0, :, :])

    ofc_bpc = s.save_and_or_load_npy_files(
        output_folder, f'th{th}_bpc.npy', lambda: s.generate_bad_pixel_corrected_array(ofc, gReconParams))

    if scan_over_a_paramter:
        r = np.linspace(-3, 3, 31)
        unit = 'mm'
        # unit = 'degrees'
        for i in trange(0, len(r)):
            centre_of_rotation_offset_x_mm = r[i]
            # centre_of_rotation_offset_y_mm = r[i]
            # gReconParams['detector_rotation'] = (math.radians(r[i]), 0., 0.)
            # gReconParams['detector_rotation'] = (0., math.radians(r[i]), 0.) # Precisely nothing!
            # gReconParams['detector_rotation'] = (0., 0., math.radians(r[i])) # Precisely nothing!

            a = 400
            img_th0, geo = s.recon_scan(gReconParams, ofc_bpc[:, a:a+11, :], angles,
                                        detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, False)
            # img_th0, geo = s.recon_scan(gReconParams, ofc_bpc[:, :, :], angles,  detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_y_mm, True)

            b = gReconParams['pixels']//2
            img = Image.fromarray(img_th0[:, :, b])
            img.save(os.path.join(output_folder,
                                  '{:s}_{:0.4f}_{:s}.tiff'.format(str(i).zfill(6), r[i], unit)))
    else:
        img_th0, geo = s.recon_scan(gReconParams, ofc_bpc, angles, detector_x_offsets,
                                    detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, True)

        ni_img = s.make_Nifti1Image(img_th0, geo.dVoxel)
        s.save_array(output_folder, f'Proj_th0_manual.nii', ni_img)
