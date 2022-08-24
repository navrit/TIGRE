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

import shared_functions as s

if __name__ == "__main__":
    '''
    1. Load 1 dataset of projections + open field images
    2. Scan over a parameter with one reconstruction each

    '''
    # Make folders and load files for recon
    drive = 'f:\\'
    base_folder = os.path.join(drive, 'jasper', 'data', '20220822_Al_Phantom_Recon_Alignment')
    base_json_file = os.path.join(base_folder, 'scan_settings.json')
    output_folder = os.path.join(base_folder, 'parameter_scanning')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ''' Load tiff files, transform a bit and save to numpy files.
    Load the files if they exist already '''
    spectral_projs_th0, spectral_open_th0, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets = s.load_or_generate_data_arrays(
        base_json_file, base_folder, output_folder)

    open_mean_th0 = np.mean(spectral_open_th0, axis=1)
    open_mean_th0[0, :, :] = open_mean_th0[0, :, :]/exp_time[0]
    spectral_projs_th0[0, :, :, :] = spectral_projs_th0[0, :, :, :] / exp_time[0]
    ofc = -np.log(spectral_projs_th0[0, :, :, :] / open_mean_th0[0, :, :])

    # Make a list of globals for the reconstruction setting, and log them in a json file
    gReconParams = dict()

    gReconParams['pixels'] = 512  # (pixels)
    gReconParams['pixel_pitch'] = 0.055  # (mm)
    gReconParams['recon_voxels'] = (
        gReconParams['pixels'], gReconParams['pixels'], gReconParams['pixels'])  # number of voxels (vx)
    gReconParams['recon_size'] = (gReconParams['pixels'] * gReconParams['pixel_pitch'], gReconParams['pixels']
                                  * gReconParams['pixel_pitch'], gReconParams['pixels'] * gReconParams['pixel_pitch'])  # (mm)

    ''' TODO These should really be read from the JSON file! '''
    gReconParams['distance_source_detector'] = 188.347  # 9+9+30+100+30+9+1.347 (mm)
    gReconParams['z_stage_distance_mm'] = 20  # Varies between 0 and 100 mm
    gReconParams['distance_object_detector'] = 30 + \
        gReconParams['z_stage_distance_mm'] + 9+1.347  # (mm)
    gReconParams['detector_rotation'] = (math.radians(-0.3), 0., 0.)  # (mm)

    centre_of_rotation_offset_y_mm = 2.51  # (mm)

    ofc_bpc = s.save_and_or_load_npy_files(
        output_folder, 'bpc.npy', lambda: s.generate_bad_pixel_corrected_array(ofc, gReconParams))

    # r = np.linspace(-1, 1, 21)
    # # unit = 'mm'
    # unit = 'degrees'
    # for i in trange(0, len(r)):
    #     gReconParams['detector_rotation'] = (math.radians(
    #         r[i]), 0., 0.)  # (mm) TODO Check accuracy!!!!!
    #     # centre_of_rotation_offset_y_mm = r[i]

    #     # a = 400
    #     # img_th0 = recon_scan(gReconParams, ofc_bpc[:, a:a+11, :], angles, z_offset,
    #     #                      detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_y_mm)

    #     img_th0 = recon_scan(gReconParams, ofc_bpc[:, :, :], angles, z_offset,
    #                          detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_y_mm)

    # ni_img = nib.Nifti1Image(img_th0, np.eye(4))
    # save_array(output_folder, f'Proj_th0_{r[i]}_{unit}.nii', ni_img)

    # b = 512//2
    # img = Image.fromarray(img_th0[:, :, b])
    # img.save(os.path.join(output_folder,
    #          '{:s}_{:0.4f}_{:s}.tiff'.format(str(i).zfill(6), r[i], unit)))

img_th0 = s.recon_scan(gReconParams, ofc_bpc[:, :, :], angles, z_offset,
                       detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_y_mm)
ni_img = nib.Nifti1Image(img_th0, np.eye(4))
s.save_array(output_folder, 'Al_Phantom.nii', ni_img)
# ni_img = nib.Nifti1Image(ofc_bpc, np.eye(4))
# save_array(output_folder, f'Proj_ofc_bpc_{0}_{unit}.nii', ni_img)

# img_th0 = recon_scan(gReconParams, ofc_bpc, angles, z_offset, detector_x_offsets,
#                         detector_y_offsets, global_detector_shift_y)

# ni_img = nib.Nifti1Image(img_th0, np.eye(4))
# save_array(output_folder, f'Proj_th0_{0}_{unit}.nii', ni_img)
