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
import SimpleITK as sitk
from tqdm import tqdm, trange


kernel = Gaussian2DKernel(x_stddev=2)

def make_gReconParams(output_folder: str, base_json_file:str) -> dict:
    assert isinstance(output_folder, str), (type(output_folder), output_folder)
    assert isinstance(base_json_file, str), (type(base_json_file), base_json_file)
    assert os.path.exists(output_folder), output_folder
    assert os.path.exists(base_json_file), base_json_file

    # Make a list of globals for the reconstruction setting, and log them in a json file
    gReconParams = dict()

    gReconParams['pixels'] = 512  # (pixels)
    gReconParams['pixel_pitch'] = 0.055  # (mm)
    gReconParams['fill_gap'] = True
    gReconParams['median_filter'] = False
    gReconParams['bad_pixel_correction'] = True
    gReconParams['recon_voxels'] = (
        gReconParams['pixels'], gReconParams['pixels'], gReconParams['pixels'])  # number of voxels (vx)
    
    '''
    9.5+9+30+100+30+9+0.055+1.035 = 188.090 mm

    9.5 mm   - Tube focal spot to tube face
    9 mm     - Edge of tube to the sample stage edge
    30 mm    - Radius of the sample stage
    100 mm   - Sample stage maximum range (variable for magnified scans - TODO implement this...)
    30 mm    - Radius of the sample stage
    9 mm     - Edge of sample stage to detector face (not mylar)
    ? mm     - Detector face to mylar - asking TODO Erik ew
    0.055 mm - Mylar cover thickness
    1.035 mm - Mylar to Si sensor surface
    '''
    gReconParams['distance_source_detector'] = 188.090
    gReconParams['z_stage_distance_mm'] = s.get_sample_z_from_first_scan_json(base_json_file) # Varies between 0 and 100 mm
    '''
    30 mm    - Radius of the sample stage
    9 mm     - Edge of sample stage to detector face (not mylar)
    ? mm     - Detector face to mylar - asking TODO Erik ew
    0.055 mm - Mylar cover thickness
    1.035 mm - Mylar to Si sensor surface
    '''
    gReconParams['distance_object_detector'] = 30 + \
        gReconParams['z_stage_distance_mm'] + 9+0.055+1.035  # (mm)
    gReconParams['detector_rotation'] = (math.radians(0.), 0., 0.)  # (mm)

    assert gReconParams['z_stage_distance_mm'] < 100 and gReconParams['z_stage_distance_mm'] >= 0

    DSD = gReconParams['distance_source_detector']
    DSO = DSD - gReconParams['distance_object_detector']

    # TODO have to explain the numbers ::: For 20220822_ffpe_whateverBreast
    a = 512 * 0.055 / (((DSD-DSO) / DSO) + 1)
    gReconParams['recon_size'] = (a, a, a)

    with open(os.path.join(output_folder, "gReconParams.json"), "w") as outfile:
        json.dump(gReconParams, outfile, indent = 4)

    return gReconParams

def load_energy_calibrations_per_pixel(folder: str, ) -> np.ndarray:
    # files = os.listdir(folder)
    # files = [x for x in os.listdir(folder) if '.npy' in os.path.splitext(x)[-1] and 'th' in os.path.splitext(x)[-1]]
    # print(files)

    files = ['th0_a.npy', 'th0_b.npy', 'th1_a.npy', 'th1_b.npy']
    slopes = np.asarray([np.load(os.path.join(folder, files[0])), np.load(os.path.join(folder, files[2]))])
    intercepts = np.asarray([np.load(os.path.join(folder, files[1])), np.load(os.path.join(folder, files[3]))])
    
    return slopes, intercepts


def generate_correct_dac_values(gReconParams: dict, open_mean_all_thr: np.ndarray, dac_list_all_chips: np.ndarray, chip_indices: list[int], slopes_per_pixel, intercepts_per_pixel, plot: bool = False, poly_order = 2, open_img_path = None) -> np.ndarray:

    def power(arr: np.ndarray, pow: int) -> np.ndarray:
        return np.array([x**pow for x in arr])
    
    def solve_poly_for_x(poly_coeffs: np.ndarray, y: float) -> np.ndarray:
        pc = poly_coeffs.copy()
        pc[-1] -= y
        return np.roots(pc)
    
    def get_chip_dac_from_array_using_orientation(dac_per_chip_list: np.ndarray, i: int, j: int, half_pixels: int, chip_indices: list[int]) -> np.ndarray:
        # assert isinstance(dac_per_chip_list, np.ndarray)
        # assert dac_per_chip_list.ndim == 2 # Always N X 4, at least 8 thresholds should have been taken...
        # assert len(dac_per_chip_list[0]) == 4 # Always 4 chips

        if (i < half_pixels and j < half_pixels):
            return dac_per_chip_list[:, chip_indices[0]]
        elif (i < half_pixels and j >= half_pixels):
            return dac_per_chip_list[:, chip_indices[1]]
        elif (i >= half_pixels and j < half_pixels):
            return dac_per_chip_list[:, chip_indices[2]]
        else:
            return dac_per_chip_list[:, chip_indices[3]]

    assert isinstance(gReconParams, dict), gReconParams
    assert isinstance(open_mean_all_thr, np.ndarray)
    assert isinstance(dac_list_all_chips, np.ndarray)
    
    assert open_mean_all_thr.ndim == 3, open_mean_all_thr.ndim  # DAC, x/y, y/x
    assert open_mean_all_thr.shape[0] > 1, open_mean_all_thr.shape
    assert open_mean_all_thr.shape[1] % 256 == 0 and open_mean_all_thr.shape[2] % 256 == 0, open_mean_all_thr.shape
    
    assert dac_list_all_chips.ndim == 2 # Always N X 4
    assert dac_list_all_chips.shape[0] >= 2 # N thresholds (DAC units) scanned over / acquired simultaneously - at least 8 should have been measured...
    assert dac_list_all_chips.shape[1] == 4 # Always 4 chips
    

    half_pixels = gReconParams['pixels'] // 2
    dac_correct = np.zeros((len(dac_list_all_chips), gReconParams['pixels'], gReconParams['pixels']))
    open_img_out = np.zeros((len(dac_list_all_chips), gReconParams['pixels'], gReconParams['pixels']), dtype=np.float32)
    # chip_indices = (0, 1, 2, 3) # TL, TR, BR, BL    Dexter = (0, 1, 2, 3)    Serval UP orientation = (2, 3, 0, 1)

    a0 = 0
    a1 = (gReconParams['pixels'] // 2) - 1 # 255 for 512 pixels
    a2 = (gReconParams['pixels'] // 2) + 1 # 257 for 512 pixels
    a3 = gReconParams['pixels']
    mean_per_dac_list = np.zeros((len(dac_list_all_chips)))

    for dac in trange(0,len(dac_list_all_chips)): # assuming that dac_list_all_chips is just for either th0 or th1
        mean_per_dac_list[dac] = 0.25 * (
            np.mean(open_mean_all_thr[dac, a0:a1, a0:a1]) +
            np.mean(open_mean_all_thr[dac, a0:a1, a2:a3]) +
            np.mean(open_mean_all_thr[dac, a2:a3, a0:a1]) +
            np.mean(open_mean_all_thr[dac, a2:a3, a2:a3])
        )   
        # print(f'DAC = {dac_list_all_chips[dac]}, mean = {mean[dac]}')

    for i in trange(0, open_mean_all_thr.shape[1]):
        for j in range(0, open_mean_all_thr.shape[2]):
            x_dacs_of_this_chip = np.asarray(get_chip_dac_from_array_using_orientation(dac_list_all_chips, i, j, half_pixels, chip_indices), dtype=np.float32)
            for th in range(2):
                x_dacs_of_this_chip[th::2] = np.add(np.multiply(slopes_per_pixel[th,i,j], x_dacs_of_this_chip[th::2]), intercepts_per_pixel[th,i,j])

            y_counts = open_mean_all_thr[:, i, j]
            assert x_dacs_of_this_chip.ndim == 1, (x_dacs_of_this_chip.ndim, x_dacs_of_this_chip)
            assert len(x_dacs_of_this_chip) == len(y_counts), (len(x_dacs_of_this_chip), len(y_counts))

            if poly_order == 0:
                raise NotImplementedError("To be implemented")
            if poly_order == 1:
                raise NotImplementedError("To be implemented")
            
            elif poly_order == 2:
                if np.isfinite(x_dacs_of_this_chip).all():
                    poly_coeffs, _, _, _, _ = np.polyfit(x_dacs_of_this_chip, y_counts, poly_order, full=True)
                    for dac in range(0, len(dac_list_all_chips)):
                        poly_roots = solve_poly_for_x(poly_coeffs, mean_per_dac_list[dac])
                        if poly_roots.shape[0] == 0:
                            dac_corr = np.NaN # x_chip_dac[dac]
                        else:
                            dac_corr = poly_roots[1]

                        dac_correct[dac, i, j] = dac_corr
                    open_img_out[:, i, j] = (poly_coeffs[0] * dac_corr**2) + (poly_coeffs[1]*dac_corr) + poly_coeffs[2]
                else:
                    open_img_out[:, i, j] = np.NaN
                
            else:
                raise NotImplementedError("To be implemented...?")

            if plot and i == 1 and j == 1:
                if poly_order == 0:
                    raise NotImplementedError
                elif poly_order == 1:
                    raise NotImplementedError
                elif poly_order == 2:
                    # Supersample so we do not get artifacts - no jagged lines, only smooth lines :)
                    fit_x = np.linspace(x_dacs_of_this_chip[0], x_dacs_of_this_chip[-1], len(x_dacs_of_this_chip))
                    y_poly_fit = (poly_coeffs[0]*power(fit_x, 2)) + \
                        (poly_coeffs[1]*power(fit_x, 1)) + poly_coeffs[2]
                    plt.xlabel('Threshold (DAC)')
                    plt.ylabel('Counts ()')
                    plt.title(f'Pixel = {i}, {j}')
                    
                    plt.errorbar(x_dacs_of_this_chip, y_counts, np.sqrt(y_counts), marker='o', label='Raw counts')
                    plt.plot(dac_correct[:, i, j], mean_per_dac_list, marker='+', markersize=10, label='Correct DAC based on open data') # zero is added for the first dac to be displayed
                    plt.plot(fit_x, y_poly_fit, label='Fit - poly order 2')

                    plt.legend()
                    plt.show()

    if open_img_path != None:
        for dac in tqdm(dac_list_all_chips):
            sitk.WriteImage(sitk.GetImageFromArray(open_img_out), os.path.join(open_img_path, f'corrected_open_img_{dac}.nii'))

    return dac_correct



def main():
    ''' TODO Looks like the data is not correctly oriented or something - completely wrong reconstructions...
        
        Per pixel energy calibrations are integrated with interleaved thresholds (th0 + th1).
        It is actually better than just using DAC units for the x axis.
    '''

    drive = 'f:\\'
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220812_BreastTissueFFPE')
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220825_LegoMan')
    base_folder = os.path.join(drive, 'jasper', 'data', '20220810_HamNCheese')

    base_json_file = os.path.join(base_folder, 'scan_settings.json')
    results_folder = os.path.join(base_folder, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    gReconParams = make_gReconParams(results_folder, base_json_file)

    # chip 0-4, th0-7 should be one of --> (4,1) (4,2) (4,8)
    # slopes = np.asarray([
    #     [.214, .222],
    #     [.205, .214],
    #     [.207, .216],
    #     [.217, .222]
    # ])
    # intercepts = np.asarray([
    #     [-1.700, -1.292],
    #     [-1.762, -1.123],
    #     [-1.439, -1.258],
    #     [-1.577, -1.210]
    # ])

    chip_indices = (3,0,1,2)

    centre_of_rotation_offset_x_mm = 0.
    centre_of_rotation_offset_y_mm = 0.
    print(f'centre_of_rotation_offset_x_mm = {centre_of_rotation_offset_x_mm} (mm)')
    print(f'centre_of_rotation_offset_y_mm = {centre_of_rotation_offset_y_mm} (mm)')

    spectral_projs_th0, spectral_open_th0, spectral_projs_th1, spectral_open_th1, th0_list, th1_list, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets, th0_dac_list, th1_dac_list = \
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
    Merging the open_means and thX_dac_list for all thresholds
    '''
    open_mean_all_thresholds = list()
    combined_x_list = list()
    
    for idx in range(len(th0_dac_list)):
        open_mean_all_thresholds.append(open_mean_th0[idx, :, :])
        open_mean_all_thresholds.append(open_mean_th1[idx, :, :])
        
        combined_x_list.append(th0_dac_list[idx])
        combined_x_list.append(th1_dac_list[idx])
        # combined_x_list.append(slopes[:, 0] * th0_dac_list[idx] + intercepts[:, 0])
        # combined_x_list.append(slopes[:, 1] * th1_dac_list[idx] + intercepts[:, 1])
    combined_x_list = np.asarray(combined_x_list)
    print(np.mean(combined_x_list, axis=1))

    open_mean_all_thresholds = np.array(open_mean_all_thresholds)

    slopes_per_pixel, intercepts_per_pixel = load_energy_calibrations_per_pixel(r'F:\Jasper\Data\EnergyCalibration\20220808')

    ofc_th0 = np.empty_like(spectral_projs_th0)
    ofc_th1 = np.empty_like(spectral_projs_th0)

    correct_dacs = s.save_and_or_load_npy_files(
        results_folder, f'all_dac_values.py', lambda: generate_correct_dac_values(gReconParams, open_mean_all_thresholds, combined_x_list, chip_indices, slopes_per_pixel, intercepts_per_pixel, plot=True, poly_order=2, open_img_path=results_folder))

    plt.imshow(correct_dacs[0, :, :])
    plt.show()
    plt.imshow(spectral_projs_th0[0, 0, :, :])
    plt.show()

    idx = 0
    for dac_index in range(0, len(combined_x_list), 2):
        for p in range(ofc_th0.shape[0]):
            ofc_th0[idx, p, :, :] = spectral_projs_th0[idx, p, :, :] - correct_dacs[dac_index, :, :]
            ofc_th0[idx, p, :, :] = -np.log(ofc_th0[idx, p, :, :] / open_mean_th0[idx, :, :])
            ofc_th1[idx, p, :, :] = spectral_projs_th1[idx, p, :, :] - correct_dacs[dac_index + 1, :, :]
            ofc_th1[idx, p, :, :] = -np.log(ofc_th1[idx, p, :, :] / open_mean_th1[idx, :, :])

        plt.imshow(ofc_th0[0, 180, :, :])
        plt.show()

        print(f'Doing recon finally! Mean energy = {combined_x_list[dac_index]} keV')
        img, geom = s.recon_scan(gReconParams, ofc_th0[idx, :, :, :], angles, detector_x_offsets,
                                 detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, True)
        ni_img = nib.Nifti1Image(img, np.eye(4))
        ni_img = s.make_Nifti1Image(img, geom.dVoxel)
        s.save_array(results_folder, 'recon_'
                     + str(combined_x_list[dac_index])+'OFC.nii', ni_img)
        idx += 1
    plt.imshow(ofc_th0[0, 0, :, :])
    plt.show()


if __name__ == "__main__":
    freeze_support()  # needed for multiprocessing on Windows - see https://stackoverflow.com/questions/63871662/python-multiprocessing-freeze-support-error
    main()
