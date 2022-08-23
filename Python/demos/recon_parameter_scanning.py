#!/usr/bin/env python
# coding: utf-8

'''
Windows: Open Anaconda prompt
conda create --name tigre_env -c anaconda -c ccpi -c conda-forge  python tigre simpleitk ipykernel opencv astropy tomopy nibabel scikit-image scikit-learn scipy tqdm scikit-learn-intelex jupyter ipywidgets
conda activate tigre_env

conda list --export > conda-package-list.txt
conda create -n tigre_env --file conda-package-list.txt
'''

import json
import math
import multiprocessing
import os
import sys
from __future__ import division


import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import tomopy
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from PIL import Image
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from skimage.registration import phase_cross_correlation
from tqdm import trange, tqdm
from typing import List

import tigre
import tigre.algorithms as algs
from tigre.utilities.geometry import Geometry

import filereader

kernel = Gaussian2DKernel(x_stddev=2)

# Make a list of globals for the reconstruction setting, and log them in a json file
gReconParams = dict()

gReconParams['pixels'] = 512  # (pixels)
gReconParams['pixel_pitch'] = 0.055  # (mm)
gReconParams['fill_gap'] = True
gReconParams['median_filter'] = False
gReconParams['bad_pixel_correction'] = True
gReconParams['recon_voxels'] = (gReconParams['pixels'], gReconParams['pixels'], gReconParams['pixels'])  # number of voxels (vx)
gReconParams['recon_size'] = (gReconParams['pixels'] * gReconParams['pixel_pitch'], gReconParams['pixels'] * gReconParams['pixel_pitch'], gReconParams['pixels'] * gReconParams['pixel_pitch'])  # (mm)

''' TODO These should really be read from the JSON file! '''
gReconParams['distance_source_detector'] = 188.347 # 9+9+30+100+30+9+1.347 (mm)
gReconParams['distance_object_detector'] = 30+ 20 +9+1.347 # (mm)
gReconParams['detector_rotation'] = (math.radians(0.), 0., 0.)  # (mm) TODO Check accuracy!!!!!


class ConeGeometryJasper(Geometry):
    ''' Some of these parameters are overwritten with better values in the recon_scan functions '''

    def __init__(self, gReconParams: dict, high_quality: bool = True, nVoxel=None):

        Geometry.__init__(self)
        pixels = gReconParams['pixels']
        pixel_pitch_mm = gReconParams['pixel_pitch']
        DSD = gReconParams['distance_source_detector']
        DOD = gReconParams['distance_object_detector']
        rotDetector = gReconParams['detector_rotation']

        total_detector_size_mm = pixels * pixel_pitch_mm

        # VARIABLE                                                DESCRIPTION                   UNITS
        # ------------------------------------------------------------------------------------------------
        if high_quality:
            self.nDetector = np.array((pixels, pixels))         # number of pixels              (px)
            self.nVoxel = np.array((pixels, pixels, pixels))    # number of voxels              (vx)
            self.sVoxel = np.array((total_detector_size_mm,
                                    total_detector_size_mm,
                                    total_detector_size_mm))    # total size of the image       (mm)
        else:
            number_of_voxels = 1
            number_of_pixels = 11

            self.nDetector = np.array((number_of_pixels,
                                       pixels))                # number of pixels              (px)
            self.nVoxel = np.array((number_of_voxels,
                                    pixels,
                                    pixels))                    # number of voxels              (vx)
            self.sVoxel = np.array((pixel_pitch_mm,
                                    pixels * pixel_pitch_mm,
                                    pixels * pixel_pitch_mm))    # total size of the image       (mm)

        ''' We will set the common variables last because other variables depend on some of them '''
        # VARIABLE                                                DESCRIPTION                   UNITS
        # ------------------------------------------------------------------------------------------------
        self.DSD = DSD                                          # Distance Source Detector      (mm)
        self.DSO = DSD - DOD                                    # Distance Source Origin        (mm)
        # Detector parameters
        self.dDetector = np.array((pixel_pitch_mm,
                                   pixel_pitch_mm))             # size of each pixel            (mm)
        self.sDetector = self.nDetector * self.dDetector        # total size of the detector    (mm)

        # Offsets
        # Offset of image from origin   (mm)
        self.offOrigin = np.array((0, 0, 0))
        self.offDetector = np.array((0, 0))                     # Offset of Detector            (mm)
        self.rotDetector = rotDetector                          # Detector rotation             (radians)
        # Image parameters
        self.dVoxel = self.sVoxel / self.nVoxel                 # size of each voxel            (mm)
        # Auxiliary
        # Accuracy of FWD proj          (vx/sample)
        self.accuracy = 0.5
        # Mode
        self.mode = 'cone'                                      # parallel, cone                ...
        self.filter = None

        '''if nVoxel is not None:
            self.DSD = 1536                                     # Distance Source Detector      (mm)
            self.DSO = 1000                                     # Distance Source Origin        (mm)
                                                                # Detector parameters
            self.nDetector = np.array((nVoxel[1],
                                       nVoxel[2])
                                                                ) # (V,U) number of pixels        (px)
            self.dDetector = np.array([0.8, 0.8])               # size of each pixel            (mm)
            self.sDetector = self.dDetector * self.nDetector    # total size of the detector    (mm)
                                                                # Image parameters
            self.nVoxel = np.array((nVoxel))                    # number of voxels              (vx)
            self.sVoxel = np.array((256, 256, 256))             # total size of the image       (mm)
            self.dVoxel = self.sVoxel / self.nVoxel             # size of each voxel            (mm)
            # Offsets
            self.offOrigin = np.array((0, 0, 0))                # Offset of image from origin   (mm)
            self.offDetector = np.array((0, 0))                 # Offset of Detector            (mm)
            self.rotDetector = np.array((0, 0, 0))
            # Auxiliary
            self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
            # Mode
            self.mode = 'cone'                                  # parallel, cone'''


def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)


def find_optimal_offset(gReconParams: dict, projections, angles, detector_x_offsets, detector_y_offsets, stage_offset=0, search_range=70):
    ''' TODO Replace this! It does not work
    
    This function takes quite a while, which can probably be optimized.
    On the other hand, we only have to do this once per scan setup, and then store the value for later use.
    '''

    projection0 = projections[0, :, :]
    projection180 = np.flip(projections[round(projections.shape[0]/2)-1, :, :], 1)

    shift, error, diffphase = phase_cross_correlation(projection0, projection180,
                                                      upsample_factor=100)

    estimate = -shift.item(1)/2
    print(estimate)
    #########################################
    # estimate = 4.9 # px
    ##########################################

    geom = ConeGeometryJasper(gReconParams, high_quality=False)
    # we should avoid pixels from the cross of the detector
    yshift = 3
    # we also need to take into account that detector motion up to 5 pixels could have been used during the acquisition
    projections_small = projections[:, int(
        projections.shape[1]/2)-6-yshift:int(projections.shape[1]/2)+5-yshift, :]

    max_value = -9999
    opt_shift = -99
    cnt = 0
    xvalues = []
    yvalues = []

    for i in trange(int(estimate*100) - search_range, int(estimate*100) + search_range, 1):
        # note that the detector x and y axis are opposite the x and y coordinate system of the geometry
        geom.offDetector = np.vstack((detector_y_offsets, detector_x_offsets+(i/100) * gReconParams['pixel_pitch'])).T
        geom.rotDetector = np.array(gReconParams['detector_rotation'])
        geom.DSD = gReconParams['distance_source_detector']
        # stageoffset = 0
        geom.DSO = geom.DSD - gReconParams['distance_object_detector'] # - stageoffset

        imgfdk = algs.fdk(projections_small, geom, angles)
        im = imgfdk[0, :, :].astype(np.float32)
        dft = cv2.dft(im, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        value = np.mean(magnitude_spectrum)
        if value > max_value:
            max_value = value
            opt_shift = i
        xvalues.append((i/100) * gReconParams['pixel_pitch'])
        yvalues.append(value)

    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.scatter(xvalues, yvalues)
    plt.show()
    return (opt_shift/100) * gReconParams['pixel_pitch']


def recon_scan(gReconParams: dict, projections, angles, sample_z_offset, detector_x_offsets, detector_y_offsets, global_detector_shift_y, high_quality=True):
    geo = ConeGeometryJasper(gReconParams, high_quality=high_quality)
    geo.offDetector = np.vstack((detector_y_offsets, detector_x_offsets+global_detector_shift_y)).T
    geo.rotDetector = np.array(gReconParams['detector_rotation'])
    geo.DSD = gReconParams['distance_source_detector']
    geo.DSO = geo.DSD - gReconParams['distance_object_detector'] - sample_z_offset

    # number of voxels              (vx)
    geo.nVoxel = np.array(gReconParams['recon_voxels'])
    geo.sVoxel = np.array(gReconParams['recon_size'])          # total size of the image       (mm)
    geo.dVoxel = geo.sVoxel/geo.nVoxel                 # size of each voxel            (mm)
    geo.offOrigin = np.array((0, 0, 0))

    # imgfdk=algs.fdk(projections,geo,angles,filter=None)
    imgfdk = algs.fdk(projections, geo, angles, filter='cosine')
    imgfdk[imgfdk < 0] = 0
    imgfdkint = (imgfdk*10000).astype(np.int32)

    imgfdkint = np.swapaxes(imgfdkint, 1, 2)
    imgfdkint = np.swapaxes(imgfdkint, 0, 2)
    imgfdkint = np.flip(imgfdkint, 2)

    return imgfdkint


def recon_scan_cgls(gReconParams: dict, projections, angles, sample_z_offset, detector_x_offsets, detector_y_offsets, global_detector_shift_y, high_quality=True):
    geo = ConeGeometryJasper(gReconParams, high_quality=high_quality)
    geo.offDetector = np.vstack((detector_y_offsets, detector_x_offsets+global_detector_shift_y)).T
    geo.rotDetector = np.array(gReconParams['detector_rotation'])
    geo.DSD = gReconParams['distance_source_detector']
    geo.DSO = geo.DSD - gReconParams['distance_object_detector'] - sample_z_offset

    # number of voxels              (vx)
    geo.nVoxel = np.array(gReconParams['recon_voxels'])
    geo.sVoxel = np.array(gReconParams['recon_size'])          # total size of the image       (mm)
    geo.dVoxel = geo.sVoxel/geo.nVoxel                 # size of each voxel            (mm)
    geo.offOrigin = np.array((0, 0, 0))
    imgfdk = algs.cgls(projections, geo, angles, 60, init='FDK')
    # imgfdk=algs.fdk(projections,geo,angles,filter=None)
    # imgfdk=algs.fdk(projections,geo,angles,filter='cosine')
    imgfdk[imgfdk < 0] = 0
    imgfdkint = (imgfdk*10000).astype(np.int32)

    imgfdkint = np.swapaxes(imgfdkint, 1, 2)
    imgfdkint = np.swapaxes(imgfdkint, 0, 2)
    imgfdkint = np.flip(imgfdkint, 2)

    return imgfdkint


def recon_scan_fista(gReconParams: dict, projections, angles, sample_z_offset, detector_x_offsets, detector_y_offsets, global_detector_shift_y, hyper, tviter, tvlambda, high_quality=True):
    geo = ConeGeometryJasper(gReconParams, high_quality=high_quality)
    geo.offDetector = np.vstack((detector_y_offsets, detector_x_offsets+global_detector_shift_y)).T
    geo.rotDetector = np.array(gReconParams['detector_rotation'])
    geo.DSD = gReconParams['distance_source_detector']
    geo.DSO = geo.DSD - gReconParams['distance_object_detector'] - sample_z_offset

    # number of voxels              (vx)
    geo.nVoxel = np.array(gReconParams['recon_voxels'])
    geo.sVoxel = np.array(gReconParams['recon_size'])          # total size of the image       (mm)
    geo.dVoxel = geo.sVoxel/geo.nVoxel                 # size of each voxel            (mm)
    geo.offOrigin = np.array((0, 0, 0))
    imgfdk = algs.fista(projections, geo, angles, 100, hyper=hyper,
                        tviter=tviter, tvlambda=tvlambda)
# )algs.cgls(projections, geo, angles, 60,init='FDK')
    # imgfdk=algs.fdk(projections,geo,angles,filter=None)
    # imgfdk=algs.fdk(projections,geo,angles,filter='cosine')
    imgfdk[imgfdk < 0] = 0
    imgfdkint = (imgfdk*10000).astype(np.int32)

    imgfdkint = np.swapaxes(imgfdkint, 1, 2)
    imgfdkint = np.swapaxes(imgfdkint, 0, 2)
    imgfdkint = np.flip(imgfdkint, 2)

    return imgfdkint


def save_array(path: str, filename: str, array: np.ndarray):
    s = os.path.join(path, filename)
    
    if (isinstance(array, np.ndarray)):
        print(f'Saving Numpy array file: {s}')
        np.save(s, array)
    elif (isinstance(array, nib.Nifti1Image)):
        print(f'Saving Nifti array file: {s}')
        nib.save(array, s)
    else:
        print('Array type supported in save_array, add it!?')


drive = 'f:\\'
# basefolder = os.path.join(drive,'jasper','data','20220726_scanseries')
# base_folder = os.path.join(drive, 'jasper', 'data', '20220812_BreastTissueFFPE')
base_folder = os.path.join(drive, 'jasper', 'data', '20220818_ffpe_WhateverBreast')
base_json_file = os.path.join(base_folder, 'scan_settings.json')
results_folder = os.path.join(base_folder, 'results_fillgap')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


''' Load tiff files, transform a bit and save to numpy files.
Load the files if they exist already '''


def files_exist(path:str, file_list:List[str]) -> bool:
    ''' If a file does not exist or is less than the threshold, return False '''
    for f in file_list:
        full_path = os.path.join(path, f)
        if not os.path.exists(full_path):
            return False
    return True

if os.path.exists(base_json_file):
    f = open(base_json_file)
    dashboard = json.load(f)
    exp_time = []
    th0_list = []
    th1_list = []

    numpy_output_files = ['projs_stack_th0.npy', 'open_stack_th0.npy',
                          'projs_stack_th1.npy', 'open_stack_th1.npy', 'thlist_th0.npy', 'thlist_th1.npy']
    if files_exist(results_folder, numpy_output_files):
        print('Loading existing numpy files, should take ~7.5 seconds')

        spectral_projs_th0 = np.load(os.path.join(results_folder, numpy_output_files[0]))
        spectral_open_th0 = np.load(os.path.join(results_folder, numpy_output_files[1]))
        spectral_projs_th1 = np.load(os.path.join(results_folder, numpy_output_files[2]))
        spectral_open_th1 = np.load(os.path.join(results_folder, numpy_output_files[3]))
        th0_list = np.load(os.path.join(results_folder, numpy_output_files[4]))
        th1_list = np.load(os.path.join(results_folder, numpy_output_files[5]))

        for i in tqdm(dashboard['thresholdscan']):
            scan_folder = os.path.join(base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
            scan_json = os.path.join(scan_folder, dashboard['thresholdscan'][i]['projections_json'])
            open_image_folder = scan_folder
            open_image_json = scan_json
            folder_string = dashboard['thresholdscan'][i]['projectionsfolder']
            
            th0_keV = folder_string[0:folder_string.find('_')]
            th1_keV = folder_string[folder_string.find('_')+1:]

            angles = filereader.get_proj_angles(scan_json)
            z_offset = filereader.get_samplestage_z_offset(scan_json)
            detector_x_offsets, detector_y_offsets = filereader.get_detector_offsets(scan_json)
            exp_time.append(filereader.get_exposure_time_projection(scan_json))
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
            scan_json = os.path.join(scan_folder, dashboard['thresholdscan'][i]['projections_json'])
            open_image_folder = scan_folder
            open_image_json = scan_json
            folder_string = dashboard['thresholdscan'][i]['projectionsfolder']
            th0_keV = folder_string[0:folder_string.find('_')]
            th1_keV = folder_string[folder_string.find('_')+1:]
            exp_time.append(filereader.get_exposure_time_projection(scan_json))
            th0_list.append(float(th0_keV))
            th1_list.append(float(th1_keV))

            projs_th0 = filereader.projectionsloader(
                scan_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
            openimg_th0 = filereader.openimgloader(
                open_image_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
            projs_th1 = filereader.projectionsloader(
                scan_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
            openimg_th1 = filereader.openimgloader(
                open_image_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
            spectral_projs_th0.append(projs_th0)
            spectral_open_th0.append(openimg_th0)
            spectral_projs_th1.append(projs_th1)
            spectral_open_th1.append(openimg_th1)
            detector_x_offsets, detector_y_offsets = filereader.get_detector_offsets(scan_json)
            angles = filereader.get_proj_angles(scan_json)
            z_offset = filereader.get_samplestage_z_offset(scan_json)
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



print(spectral_open_th0.shape)  # E.g. (9, 32, 512, 512) (Thresholds, # of open images, pixels, pixels)

open_mean_th0 = np.mean(spectral_open_th0, axis=1)
open_mean_th1 = np.mean(spectral_open_th1, axis=1)

for i in range(open_mean_th0.shape[0]):
    # print(i, open_mean_th0.shape, open_mean_th0.shape[0])  # E.g. 0 (9, 512, 512) 9
    open_mean_th0[i, :, :] = open_mean_th0[i, :, :]/exp_time[i]
    open_mean_th1[i, :, :] = open_mean_th1[i, :, :]/exp_time[i]

for i in range(spectral_projs_th0.shape[0]):
    spectral_projs_th0[i, :, :, :] = spectral_projs_th0[i, :, :, :] / exp_time[i]
    spectral_projs_th1[i, :, :, :] = spectral_projs_th1[i, :, :, :] / exp_time[i]


for th in range(0, open_mean_th0.shape[0]):
    a0 = 0
    a1 = (gReconParams['pixels'] // 2) - 1 # 255 for 512 pixels
    a2 = (gReconParams['pixels'] // 2) + 1 # 257 for 512 pixels
    a3 = gReconParams['pixels']
    mean = 0.25 * (
            np.mean(open_mean_th0[th, a0:a1, a0:a1]) +
            np.mean(open_mean_th0[th, a0:a1, a2:a3]) +
            np.mean(open_mean_th0[th, a2:a3, a0:a1]) +
            np.mean(open_mean_th0[th, a2:a3, a2:a3])
    )
    print(f'th = {th}, mean = {mean}')  # E.g. 36895.64768079413
    regression_out = np.zeros((3, gReconParams['pixels'], gReconParams['pixels']))
    DAC_values = np.zeros((gReconParams['pixels'], gReconParams['pixels']))

    print('Finding best DAC values per pixel...')
    for i in trange(0, open_mean_th0.shape[1]):
        for j in range(0, open_mean_th0.shape[2]):
            yvalues = open_mean_th0[:, i, j]
            regressions, res, _, _, _ = np.polyfit(th0_list, yvalues, 2, full=True)
            regression_out[:, i, j] = regressions
            DAC = solve_for_y(regressions, mean)[1]
            if (DAC > th0_list[th]*2) or (DAC < th0_list[th]/2):
                DAC = th0_list[th]
            DAC_values[i, j] = DAC
    
    fit_array = spectral_projs_th0.reshape(spectral_projs_th0.shape[0], -1)

    print('Fitting polynomials... ~1 minute')
    regressions, res, _, _, _ = np.polyfit(th0_list, fit_array, 2, full=True)

    print('Calculating DACs...')
    calc_dacs = np.repeat(DAC_values[np.newaxis, :, :], spectral_projs_th0.shape[1], axis=0).flatten()

    print('Calculating projection data...')
    proj_data_flat = (regressions[0, :]*calc_dacs**2) + \
        (regressions[1, :]*calc_dacs**1) + regressions[2, :]
    projection_data = proj_data_flat.reshape(
        spectral_projs_th0.shape[1], spectral_projs_th0.shape[2], spectral_projs_th0.shape[3])
    save_array(results_folder, 'Projections_th0_' + str(th0_list[th])+'_keV.npy', projection_data)
    
    # TODO Ask Jasper about this VERY SUSPICIOUS CODE
    mean = (np.mean(projection_data[:, :, 0:10]) + np.mean(projection_data[:, :, 503:513]))/2
    ofc = -np.log(projection_data/mean)
    save_array(results_folder, 'projs_th0_'+str(th0_list[th])+'OFC_interp.npy', ofc)

    print('Doing median filter on OFC data...')
    ofc_mf = filereader.median_filter_projection_set(ofc, 5)
    diff_mf = np.abs(ofc-ofc_mf)
    meanmap = np.mean(diff_mf, axis=0)
    stdmap = np.std(diff_mf, axis=0)
    badmap = np.ones((gReconParams['pixels'], gReconParams['pixels']))
    half = np.int32(badmap.shape[0]/2)
    badmap[half-2:half+1] = 0
    badmap[:, half-2:half+1] = 0
    badmap[meanmap > 0.2] = 0
    badmap[stdmap > 0.05] = 0
    ofc_bpc = filereader.apply_badmap_to_projections(ofc, badmap)
    ofc_bpc_mf = filereader.median_filter_projection_set(ofc_bpc, 3)
    if th == 0:
        # print(f'th = {th}, finding optimal offset')
        global_detector_shift_y = 0.25 # (mm) # find_optimal_offset(gReconParams, spectral_projs_th0[1, :, :, :], angles, detector_x_offsets, detector_y_offsets, stage_offset=0, search_range=25)
        print(f'global_detector_shift_y = {global_detector_shift_y} (mm)')
        
        ni_img = nib.Nifti1Image(ofc_bpc_mf, np.eye(4))
        save_array(results_folder, 'Proj_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)
    
    print('Doing recon finally!')
    img_th0 = recon_scan(gReconParams, ofc_bpc, angles, z_offset, detector_x_offsets,
                         detector_y_offsets, global_detector_shift_y)
    
    ni_img = nib.Nifti1Image(img_th0, np.eye(4))
    save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC.nii', ni_img)

    img_th0 = recon_scan(gReconParams, ofc_bpc_mf, angles, z_offset, detector_x_offsets,
                         detector_y_offsets, global_detector_shift_y)
    ni_img = nib.Nifti1Image(img_th0, np.eye(4))
    save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)



# print(f'th = {th}, finding optimal offset')
# global_detector_shift_y = .2 #find_optimal_offset(spectral_projs_th0[1, :, :, :], angles, detector_x_offsets, detector_y_offsets, stageoffset=0, searchrange=25)
# print(f'global_detector_shift_y = {global_detector_shift_y} (mm)')

# ni_img = nib.Nifti1Image(ofc_bpc_mf, np.eye(4))
# save_array(results_folder, 'Proj_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)

# print('Doing recon finally!')
# img_th0 = recon_scan(gReconParams, ofc_bpc, angles, z_offset, detector_x_offsets,
#                     detector_y_offsets, global_detector_shift_y)

# ni_img = nib.Nifti1Image(img_th0, np.eye(4))
# save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC.nii', ni_img)

# img_th0 = recon_scan(gReconParams, ofc_bpc_mf, angles, z_offset, detector_x_offsets,
#                     detector_y_offsets, global_detector_shift_y)
# ni_img = nib.Nifti1Image(img_th0, np.eye(4))
# save_array(results_folder, 'Recon_th0_'+str(th0_list[th])+'OFC_BPC_MF.nii', ni_img)
