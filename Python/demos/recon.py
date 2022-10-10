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
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from astropy.convolution import Gaussian2DKernel
from lmfit.models import GaussianModel, StepModel
from lmfit.lineshapes import gaussian, step
from scipy.interpolate import interp1d, UnivariateSpline
from tqdm import tqdm, trange
from joblib import Parallel, delayed

import shared_functions as s

kernel = Gaussian2DKernel(x_stddev=2)


def make_gReconParams(output_folder: str, base_json_file: str) -> dict:
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
    gReconParams['z_stage_distance_mm'] = s.get_sample_z_from_first_scan_json(
        base_json_file)  # Varies between 0 and 100 mm
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
        json.dump(gReconParams, outfile, indent=4)

    return gReconParams


def load_energy_calibrations_per_pixel(folder: str, ) -> np.ndarray:
    # files = os.listdir(folder)
    # files = [x for x in os.listdir(folder) if '.npy' in os.path.splitext(x)[-1] and 'th' in os.path.splitext(x)[-1]]
    # print(files)

    files = ['th0_a.npy', 'th0_b.npy', 'th1_a.npy', 'th1_b.npy']
    slopes = np.asarray([np.load(os.path.join(folder, files[0])),
                        np.load(os.path.join(folder, files[2]))])
    intercepts = np.asarray([np.load(os.path.join(folder, files[1])),
                            np.load(os.path.join(folder, files[3]))])

    return slopes, intercepts


def get_chip_dac_from_array_using_orientation(dac_per_chip_list: np.ndarray, i: int, j: int, half_pixels: int, chip_indices: List[int]) -> np.ndarray:
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


def gaussian_fit(x, y):
    model = GaussianModel()
    parameters = model.guess(y, x=x)  # TODO could maybe move this outside the loop?
    out_gaussian = model.fit(y, parameters, x=x)
    return out_gaussian.best_values['amplitude'], out_gaussian.best_values['center'], out_gaussian.best_values['sigma']


def fit_proj_data_values_gaussian(data: np.ndarray, dacs_list: np.ndarray) -> np.ndarray:
    assert data.ndim == 4  # DAC, proj, y/x, y/x
    assert data.shape[0] >= 3  # We need at least 3 points to do a fit...
    # assert data.shape[1] > 100  # Should be >100 projections?
    # assert data.shape[2] % 128 == 0 # Should be 128 or 256 (one chip in FPM or colour mode)
    assert data.shape[2] == data.shape[3]  # Should be a square

    fit_array = data.reshape(data.shape[0], -1)  # Flatten and transpose

    # fit
    parameters = 3
    y_gaussian_parameters = np.empty(
        shape=(data.shape[1] * data.shape[2] * data.shape[3]))
    amplitudes = np.empty_like(fit_array[0])
    centres = np.empty_like(amplitudes)
    sigmas = np.empty_like(amplitudes)

    # for i in trange(0, len(y_gaussian_parameters), parameters):
    #     amplitudes[i], centres[i], sigmas[i] = gaussian_fit(
    #         model_gaussian, dacs_list, fit_array[:, i])
    y_gaussian_parameters = np.asarray(Parallel(n_jobs=32)(delayed(gaussian_fit)(dacs_list, fit_array[:,i]) for i in trange(0, data.shape[1] * data.shape[2] * data.shape[3], 1)))
    amplitudes = y_gaussian_parameters[:,0]
    centres = y_gaussian_parameters[:,1]
    sigmas = y_gaussian_parameters[:,2]

    amplitudes = amplitudes.reshape(data.shape[1], data.shape[2], data.shape[3])
    centres = centres.reshape(data.shape[1], data.shape[2], data.shape[3])
    sigmas = sigmas.reshape(data.shape[1], data.shape[2], data.shape[3])

    return amplitudes, centres, sigmas


def recon_using_gaussian_fit(dac_list_all_chips, i, j, half_pixels, chip_indices, y_counts, model_gaussian, fit_x):
    # assert x_dacs_of_this_chip.ndim == 1, (x_dacs_of_this_chip.ndim, x_dacs_of_this_chip)
    # assert len(x_dacs_of_this_chip) == len(y_counts), (len(x_dacs_of_this_chip), len(y_counts))

    x_dacs_of_this_chip = get_chip_dac_from_array_using_orientation(
        dac_list_all_chips, i, j, half_pixels, chip_indices)
    parameters = gaussian_fit(model_gaussian, x_dacs_of_this_chip, y_counts)
    y_gaussian = gaussian(fit_x, parameters[0], parameters[1], parameters[2])

    try:
        # dac_correct[:, i, j] = f(y_counts) # map back from y to x coords
        # open_img_out[:, i, j] = y_gaussian
        # Could do fill_value='extrapolate'
        f = interp1d(y_gaussian, fit_x, bounds_error=None, fill_value=np.nan)
        return f(y_counts)
    except ValueError:
        return np.repeat(np.nan, len(y_counts))


def extract_results_into_numpy_array(_data, _data_size: int, _n: int):
    assert isinstance(_data_size, int)
    assert isinstance(_n, int)

    array = np.empty(shape=[_n, _data_size, _data_size], dtype=np.float64)
    for p in trange(_n):
        for i in range(_data_size**2):
            array[p, int(i/_data_size), i % _data_size] = _data[i][p]

    return array


def generate_correct_dac_values(gReconParams: dict, open_median_all_thr: np.ndarray, dac_list_all_chips: np.ndarray, chip_indices: List[int], plot: bool = False, open_img_path=None) -> np.ndarray:
    assert isinstance(gReconParams, dict), gReconParams
    assert isinstance(open_median_all_thr, np.ndarray)
    assert isinstance(dac_list_all_chips, np.ndarray)

    assert open_median_all_thr.ndim == 3, open_median_all_thr.ndim  # DAC, x/y, y/x
    assert open_median_all_thr.shape[0] > 1, open_median_all_thr.shape
    assert open_median_all_thr.shape[1] % 256 == 0 and open_median_all_thr.shape[2] % 256 == 0, open_median_all_thr.shape

    assert dac_list_all_chips.ndim == 2  # Always N X 4
    # N thresholds (DAC units) scanned over / acquired simultaneously - at least 8 should have been measured...
    assert dac_list_all_chips.shape[0] >= 2
    assert dac_list_all_chips.shape[1] == 4  # Always 4 chips

    half_pixels = gReconParams['pixels'] // 2
    fit_x = np.linspace(10, 100, 91)  # x_dacs_of_this_chip[0]-20, x_dacs_of_this_chip[-1]+20
    dac_correct = np.zeros(
        (len(dac_list_all_chips), gReconParams['pixels'], gReconParams['pixels']))
    # open_img_out = np.zeros((len(fit_x), gReconParams['pixels'], gReconParams['pixels']), dtype=np.float32)
    # chip_indices = (0, 1, 2, 3) # TL, TR, BR, BL    Dexter = (0, 1, 2, 3)    Serval UP orientation = (2, 3, 0, 1)    Serval UP MIRRORED orientation = (3, 0, 1, 2)

    a0 = 0
    a1 = (gReconParams['pixels'] // 2) - 1  # 255 for 512 pixels
    a2 = (gReconParams['pixels'] // 2) + 1  # 257 for 512 pixels
    a3 = gReconParams['pixels']
    median_per_dac_list = np.zeros((len(dac_list_all_chips)))

    # assuming that dac_list_all_chips is just for either th0 or th1
    for dac in trange(0, len(dac_list_all_chips)):
        median_per_dac_list[dac] = 0.25 * (
            np.median(open_median_all_thr[dac, a0:a1, a0:a1])
            + np.median(open_median_all_thr[dac, a0:a1, a2:a3])
            + np.median(open_median_all_thr[dac, a2:a3, a0:a1])
            + np.median(open_median_all_thr[dac, a2:a3, a2:a3])
        )
        # print(f'DAC = {dac_list_all_chips[dac]}, mean = {mean[dac]}')

    model_gaussian = GaussianModel()
    # mod_erf = StepModel(form='erf')

    N = 1
    results = Parallel(n_jobs=60)(delayed(recon_using_gaussian_fit)(dac_list_all_chips, i, j, half_pixels, chip_indices,
                                                                    open_median_all_thr[:, i, j], model_gaussian, fit_x) for i in trange(open_median_all_thr.shape[1] // N) for j in range(open_median_all_thr.shape[2] // N))
    dac_correct = extract_results_into_numpy_array(
        results, gReconParams['pixels']//N, len(dac_list_all_chips))

    plt.imshow(dac_correct[0], vmin=np.nanmin(dac_correct[0]), vmax=np.nanmax(dac_correct[0]))
    plt.show()
    plt.imshow(dac_correct[-1], vmin=np.nanmin(dac_correct[-1]), vmax=np.nanmax(dac_correct[-1]))
    plt.show()

    # for i in trange(open_median_all_thr.shape[1]): # open_median_all_thr.shape[1]
    #     for j in range(open_median_all_thr.shape[2]): # open_median_all_thr.shape[2]
    #         x_dacs_of_this_chip = get_chip_dac_from_array_using_orientation(dac_list_all_chips, i, j, half_pixels, chip_indices)

    #         y_counts = open_median_all_thr[:, i, j]
    #         # assert x_dacs_of_this_chip.ndim == 1, (x_dacs_of_this_chip.ndim, x_dacs_of_this_chip)
    #         # assert len(x_dacs_of_this_chip) == len(y_counts), (len(x_dacs_of_this_chip), len(y_counts))

    #         pars_gaussian = model_gaussian.guess(y_counts, x=x_dacs_of_this_chip) # TODO could maybe move this outside the loop?
    #         out_gaussian = model_gaussian.fit(y_counts, pars_gaussian, x=x_dacs_of_this_chip)
    #         y_gaussian = gaussian(fit_x, out_gaussian.best_values['amplitude'], out_gaussian.best_values['center'], out_gaussian.best_values['sigma'])

    #         f = interp1d(y_gaussian, fit_x, bounds_error=False, fill_value=np.NaN) # Default: fill_value=np.nan  could do fill_value='extrapolate'
    #         dac_correct[:, i, j] = f(y_counts) # map back from y to x coords
    #         open_img_out[:, i, j] = y_gaussian

    #         if plot and i == 0: #and j == 1:
    #             # y_counts_flip = np.flip(y_counts)
    #             # pars_orig_erf = mod_erf.guess(y_counts_flip, x=x_dacs_of_this_chip)
    #             # out_erf = mod_erf.fit(y_counts_flip, pars_orig_erf, x=x_dacs_of_this_chip)

    #             # spl_default_smoothing = UnivariateSpline(x_dacs_of_this_chip, y_counts, ext='extrapolate', k=1)

    #             # y_erf = step(fit_x, out_erf.best_values['amplitude'], out_erf.best_values['center'], out_erf.best_values['sigma'], form='erf')[::-1]
    #             # y_spline = spl_default_smoothing(fit_x)
    #             plt.xlabel('Threshold (DAC)')
    #             plt.ylabel('Counts ()')
    #             plt.title(f'Pixel = {i}, {j}')

    #             plt.errorbar(x_dacs_of_this_chip, y_counts, np.sqrt(y_counts), marker='o', label='Raw counts')
    #             plt.plot(dac_correct[:, i, j], median_per_dac_list, marker='+', markersize=10, label='Correct DAC based on median air data')

    #             plt.plot(fit_x, y_gaussian, label='Fit - Gaussian')
    #             # plt.plot(fit_x, y_erf, label='Fit - 1-erf')
    #             # plt.plot(fit_x, y_spline, label='Fit - Spline')

    #             plt.legend()
    #             plt.show()

    # if open_img_path != None:
    #     for dac in tqdm(dac_list_all_chips):
    #         sitk.WriteImage(sitk.GetImageFromArray(open_img_out), os.path.join(open_img_path, f'corrected_open_img_{dac}.nii'))

    return dac_correct


def main():
    drive = 'f:\\'
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220812_BreastTissueFFPE')
    # base_folder = os.path.join(drive, 'jasper', 'data', '20220825_LegoMan')
    base_folder = os.path.join(drive, 'jasper', 'data', '20220810_HamNCheese')

    # drive = 'D:\\'
    # base_folder = os.path.join(drive, '20220810_HamNCheese')

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

    chip_indices = (3, 0, 1, 2)
    # p0 = 0
    # p1 = gReconParams['pixels']//2
    # p2 = gReconParams['pixels']
    # chip_coords = ((p0,p1, p0,p1), (p0,p1, p1,p2), (p1,p2, p1,p2), (p1,p2, p0,p1)) # TL, TR, BR, BL  (coord_start, coord_end)

    centre_of_rotation_offset_x_mm = 0.
    centre_of_rotation_offset_y_mm = 0.
    print(f'centre_of_rotation_offset_x_mm = {centre_of_rotation_offset_x_mm} (mm)')
    print(f'centre_of_rotation_offset_y_mm = {centre_of_rotation_offset_y_mm} (mm)')

    spectral_projs_th0, spectral_open_th0, spectral_projs_th1, spectral_open_th1, th0_list, th1_list, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets, th0_dacs_list, th1_dacs_list = \
        s.load_or_generate_data_arrays(base_json_file, base_folder, results_folder, gReconParams)

    # E.g. (9, 32, 512, 512) (Thresholds, # of open images, y, x)
    print(spectral_open_th0.shape)

    open_median_th0 = np.median(spectral_open_th0, axis=1)
    open_median_th1 = np.median(spectral_open_th1, axis=1)

    for i in range(open_median_th0.shape[0]):
        # print(i, open_median_th0.shape, open_median_th0.shape[0])  # E.g. 0 (9, 512, 512) 9
        open_median_th0[i, :, :] = open_median_th0[i, :, :]/exp_time[i]
        open_median_th1[i, :, :] = open_median_th1[i, :, :]/exp_time[i]

    for i in range(spectral_projs_th0.shape[0]):
        spectral_projs_th0[i, :, :, :] = spectral_projs_th0[i, :, :, :] / exp_time[i]
        spectral_projs_th1[i, :, :, :] = spectral_projs_th1[i, :, :, :] / exp_time[i]

    ofc_th0 = np.empty_like(spectral_projs_th0)
    # ofc_th1 = np.empty_like(spectral_projs_th0)
    # ofc_bpc_th0 = np.empty_like(ofc_th0)

    corrected_DAC_values_from_open_images_th0 = s.save_and_or_load_npy_files(
        results_folder, f'th0_dac_values.npy', lambda: generate_correct_dac_values(gReconParams, open_median_th0, th0_dacs_list, chip_indices, plot=False, open_img_path=results_folder))
    # corrected_DAC_values_from_open_images_th1 = s.save_and_or_load_npy_files(
    #     results_folder, f'th1_dac_values.npy', lambda: generate_correct_dac_values(gReconParams, open_median_th1, th1_dac_list, chip_indices, plot=False, open_img_path=results_folder))

    ''' With these corrected DACs we use the Gaussian (in this case) fits per pixel on the projection data to calculate the corrected count data. 
    
    TODO I could use spline/interpolated data also! '''

    N = spectral_projs_th0.shape[1]
    p0 = 0
    p1 = gReconParams['pixels']//2
    p2 = p1 * 2

    ''' TODO Refactor and test this so it runs per image, not on the whole 4D array / pixels simultaneously. '''
    # spectral_projs_th0 --> Points on x axis (scanned DAC or similar) projections y x
    regressions_2 = fit_proj_data_values_gaussian(spectral_projs_th0[:, :N, p1:p2, p1:p2], th0_dacs_list[:, 2])  # BR
    regressions_1 = fit_proj_data_values_gaussian(spectral_projs_th0[:, :N, p1:p2, p0:p1], th0_dacs_list[:, 1])  # BL
    regressions_3 = fit_proj_data_values_gaussian(spectral_projs_th0[:, :N, p0:p1, p0:p1], th0_dacs_list[:, 3])  # TL
    regressions_0 = fit_proj_data_values_gaussian(spectral_projs_th0[:, :N, p0:p1, p1:p2], th0_dacs_list[:, 0])  # TR

    # plt.plot(np.nanmedian(regressions_0[0], axis=(1, 2)))
    # plt.show()
    # plt.plot(np.nanmedian(regressions_0[1], axis=(1, 2)))
    # plt.show()
    # plt.plot(np.nanmedian(regressions_0[2], axis=(1, 2)))
    # plt.show()
    # print(np.nanmedian(regressions_0[0], axis=(1, 2)))
    # print(np.nanmedian(regressions_0[1], axis=(1, 2)))
    # print(np.nanmedian(regressions_0[2], axis=(1, 2)))

    proj_data_fits = np.empty(
        (3, spectral_projs_th0.shape[1], spectral_projs_th0.shape[2], spectral_projs_th0.shape[3]))

    proj_data_fits[:, :N, p1:p2, p1:p2] = regressions_0
    proj_data_fits[:, :N, p1:p2, p0:p1] = regressions_1
    proj_data_fits[:, :N, p0:p1, p0:p1] = regressions_2
    proj_data_fits[:, :N, p0:p1, p1:p2] = regressions_3

    s.save_array(results_folder, 'proj_data_fits.npy', proj_data_fits)

    ''' Does this selected region contain only air over the whole scan?
        TODO Should do this per sinogram row (per time / projection) '''
    air_column_number = 5

    # img_check = np.median(spectral_projs_th0[0]/np.sum(spectral_open_th0[0], axis=0), axis=1)
    # plt.imshow(img_check)
    # plt.axvline(0, c='k')
    # plt.axvline(air_column_number, c='r')
    # plt.axvline(gReconParams['pixels'] - air_column_number, c='r')
    # plt.axvline(gReconParams['pixels'], c='k')
    # plt.show()

    idx = 0
    for dac_index in trange(0, spectral_projs_th0.shape[0]):
        d = corrected_DAC_values_from_open_images_th0[dac_index]  # (512, 512)
        corrected_dacs_from_open_images_for_n_projections = np.repeat(
            d[np.newaxis, :, :], spectral_projs_th0.shape[1], axis=0).flatten()  # (Projections, x/y, y/x) FLATTENED
        proj_data_fits_flat = proj_data_fits.reshape(
            proj_data_fits.shape[0], -1)  # Transpose and flatten projection data

        # reconstructed_proj_data_flat = np.load(os.path.join(results_folder, 'Projections_th0_' + str(th0_list[dac_index])+'_DAC.npy'))

        reconstructed_proj_data_flat = np.empty_like(
            corrected_dacs_from_open_images_for_n_projections)

        i = 0
        a = corrected_DAC_values_from_open_images_th0[:, 0, 0]
        b = proj_data_fits_flat[0, i]
        c = proj_data_fits_flat[1, i]
        d = proj_data_fits_flat[2, i]
        e = gaussian(a, b, c, d)
        # plt.plot(a, e)
        # plt.show()
        # reconstructed_proj_data_flat[i:ii] = gaussian(corrected_dacs_from_open_images_for_n_projections[i:ii], proj_data_fits_flat[0, i:ii], proj_data_fits_flat[1, i:ii], proj_data_fits_flat[2, i:ii])
        # for i in trange(corrected_dacs_from_open_images_for_n_projections.shape[0]):
        #     reconstructed_proj_data_flat[i] = gaussian(corrected_dacs_from_open_images_for_n_projections[i], proj_data_fits_flat[0, i], proj_data_fits_flat[1, i], proj_data_fits_flat[2, i])
        # reconstructed_proj_data_flat = np.asarray(Parallel(n_jobs=60)(delayed(gaussian)(corrected_dacs_from_open_images_for_n_projections[i], proj_data_fits_flat[0, i], proj_data_fits_flat[1, i], proj_data_fits_flat[2, i]) for i in trange(corrected_dacs_from_open_images_for_n_projections.shape[0])))

        corrected_projection_data_th0 = reconstructed_proj_data_flat.reshape(
            spectral_projs_th0.shape[1], spectral_projs_th0.shape[2], spectral_projs_th0.shape[3])
        s.save_array(results_folder, 'Projections_th0_'
                     + str(th0_list[dac_index])+'_DAC.npy', corrected_projection_data_th0)

        median = (np.nanmedian(corrected_projection_data_th0[:, :, 0:air_column_number]) + np.nanmedian(
            corrected_projection_data_th0[:, :, gReconParams['pixels']-air_column_number:gReconParams['pixels']]))/2
        ofc_th0 = -np.log(corrected_projection_data_th0/median)
        # ofc_bpc_th0 = s.save_and_or_load_npy_files(results_folder, f'th{dac_index}_bpc.npy', lambda: s.generate_bad_pixel_corrected_array(ofc_th0, gReconParams))

        # plt.imshow(ofc_th0[0, :, :])
        # plt.show()

        print(f'Doing recon finally! Mean energy = {th0_dacs_list[dac_index]} keV')
        img, geom = s.recon_scan(gReconParams, ofc_th0, angles, detector_x_offsets,
                                 detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, True)
        ni_img = nib.Nifti1Image(img, np.eye(4))
        ni_img = s.make_Nifti1Image(img, geom.dVoxel)
        s.save_array(results_folder, 'recon_'
                     + str(th0_dacs_list[dac_index])+'OFC.nii', ni_img)
        idx += 1


if __name__ == "__main__":
    freeze_support()  # needed for multiprocessing on Windows - see https://stackoverflow.com/questions/63871662/python-multiprocessing-freeze-support-error
    main()
