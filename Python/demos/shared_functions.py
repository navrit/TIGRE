import json
import multiprocessing
import os
from multiprocessing import freeze_support
from typing import List

import nibabel as nib
import numpy as np
import tigre.algorithms as algs
from PIL import Image
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from tigre.utilities.geometry import Geometry
from tqdm import tqdm, trange
from skimage.registration import phase_cross_correlation
import cv2
from joblib import Parallel, delayed
import SimpleITK as sitk

import matplotlib.pyplot as plt


def load_projection(filepaths, badpixelcorr=True, medianfilter=False, fillgap=False):
    finalarray = np.array([])
    for filepath in filepaths:
        if os.path.exists(filepath):
            imarray = np.array(Image.open(os.path.join(filepath))).astype(np.float64)
            if not np.any(finalarray):
                finalarray = imarray
            else:
                finalarray = finalarray + imarray

    if np.any(finalarray):
        # if fillgap:
        #     gap = 2
        #     cross = np.zeros((finalarray.shape[0]+gap, finalarray.shape[1]+gap))
        #     half = np.int32(finalarray.shape[0]/2)
        #     cross[0:half, 0:half] = finalarray[0:half, 0:half]
        #     cross[0:half, half+gap:cross.shape[1]] = finalarray[0:half, half:finalarray.shape[1]]
        #     cross[half+gap:cross.shape[0], 0:half] = finalarray[half:finalarray.shape[0], 0:half]
        #     cross[half+gap:cross.shape[0], half+gap:cross.shape[1]
        #           ] = finalarray[half:finalarray.shape[0], half:finalarray.shape[1]]
        #     for j in range(0, cross.shape[0]):
        #         value = cross[j, half-1]/2
        #         cross[j, half-1:half+gap] = value
        #         value = cross[j, half+gap]/2
        #         cross[j, half-1+gap:half+1+gap] = value
        #         value = cross[half-1, j]/2
        #         cross[half-1:half+gap, j] = value
        #         value = cross[half+gap, j]/2
        #         cross[half-1+gap:half+1+gap, j] = value
        #     cross[256:258, 256:258] = np.nan
        #     finalarray = cross[1:-1, 1:-1]
        if fillgap:
            gap = 4
            cross = np.zeros((finalarray.shape[0]+gap, finalarray.shape[1]+gap))
            half = np.int32(finalarray.shape[0]/2)
            cross[0:half, 0:half] = finalarray[0:half, 0:half]
            cross[0:half, half+gap:cross.shape[1]] = finalarray[0:half, half:finalarray.shape[1]]
            cross[half+gap:cross.shape[0], 0:half] = finalarray[half:finalarray.shape[0], 0:half]
            cross[half+gap:cross.shape[0], half+gap:cross.shape[1]
                  ] = finalarray[half:finalarray.shape[0], half:finalarray.shape[1]]
            for j in range(0, cross.shape[0]):
                value = cross[j, half-1]/2
                cross[j, half-1:half+gap] = value
                value = cross[j, half+gap]/2
                cross[j, half-1+gap:half+1+gap] = value
                value = cross[half-1, j]/2
                cross[half-1:half+gap, j] = value
                value = cross[half+gap, j]/2
                cross[half-1+gap:half+1+gap, j] = value
            cross[half:half+gap, half:half+gap] = np.nan
            finalarray = cross[2:-2, 2:-2]

        if badpixelcorr:
            finalarray[finalarray == np.inf] = 0
            finalarray[finalarray == -np.inf] = 0
            finalarray[finalarray > 1e6] = 0
            finalarray[finalarray == 0] = 0
            half = np.int32(finalarray.shape[0]/2)
            finalarray[half-1] = 0
            finalarray[half] = 0
            finalarray[:, half-1] = 0
            finalarray[:, half] = 0
            if np.count_nonzero(finalarray == 0) > 0:
                valid_mask = (finalarray > 0)
                coords = np.array(np.nonzero(valid_mask)).T
                values = finalarray[valid_mask]
                it = interpolate.NearestNDInterpolator(coords, values)
                finalarray = it(list(np.ndindex(finalarray.shape))).reshape(finalarray.shape)
            # finalarray = interpolate_replace_nans(finalarray, kernel)#.astype(np.int16)

        if medianfilter:
            finalarray = median_filter(finalarray, 5)

    return finalarray


def projectionsloader(jsonfile='', th0=True, badpixelcorr=True, medianfilter=False, fillgap=False):
    freeze_support()

    if os.path.exists(jsonfile):
        dirname = os.path.dirname(jsonfile)
        f = open(jsonfile)
        dashboard = json.load(f)
        totalfilelist = []
        if th0:
            for i in dashboard['projections']:
                projectionlist = []
                for j in dashboard['projections'][i]['filenames_th0']:
                    filename = os.path.join(
                        dirname, dashboard['projections'][i]['filenames_th0'][j]['name'])
                    projectionlist.append(filename)
                totalfilelist.append(projectionlist)
        else:

            for i in dashboard['projections']:
                projectionlist = []
                for j in dashboard['projections'][i]['filenames_th1']:
                    filename = os.path.join(
                        dirname, dashboard['projections'][i]['filenames_th1'][j]['name'])
                    projectionlist.append(filename)
                totalfilelist.append(projectionlist)
        f.close()
        with multiprocessing.Pool(60) as pool:
            processes = [pool.apply_async(load_projection, args=(
                x, badpixelcorr, medianfilter, fillgap)) for x in totalfilelist]
            result = [p.get() for p in processes]
            return np.asarray(result)
    else:
        return None


def openimgloader(jsonfile='', th0=True, badpixelcorr=True, medianfilter=False, fillgap=False):
    freeze_support()

    if os.path.exists(jsonfile):
        dirname = os.path.dirname(jsonfile)

        f = open(jsonfile)
        dashboard = json.load(f)
        totalfilelist = []
        if th0:
            for i in dashboard['openimages']:
                projectionlist = []
                for j in dashboard['openimages'][i]['filenames_th0']:
                    filename = os.path.join(
                        dirname, dashboard['openimages'][i]['filenames_th0'][j]['name'])
                    projectionlist.append(filename)
                totalfilelist.append(projectionlist)
        else:
            for i in dashboard['openimages']:
                projectionlist = []
                for j in dashboard['openimages'][i]['filenames_th1']:
                    filename = os.path.join(
                        dirname, dashboard['openimages'][i]['filenames_th1'][j]['name'])
                    projectionlist.append(filename)
                totalfilelist.append(projectionlist)
        f.close()

        with multiprocessing.Pool(60) as pool:
            processes = [pool.apply_async(load_projection, args=(
                x, badpixelcorr, medianfilter, fillgap)) for x in totalfilelist]
            result = [p.get() for p in processes]
            return np.asarray(result)

    else:
        return None


def get_sample_z_from_first_scan_json(json_file_full_path: str) -> float:
    if os.path.exists(json_file_full_path):
        with open(json_file_full_path) as f:
            j = json.load(f)

            base_path = os.path.dirname(json_file_full_path)
            projections_folder = j["thresholdscan"]["0000"]["projectionsfolder"]
            projections_json = j["thresholdscan"]["0000"]["projections_json"]
            sub_json = os.path.join(base_path, projections_folder, projections_json)

            if os.path.exists(sub_json):
                with open(sub_json) as f:
                    j = json.load(f)
                    return j['projections']["0000"]['sample_z']
    else:
        return None


def get_proj_angles(jsonfile=''):
    if os.path.exists(jsonfile):
        f = open(jsonfile)
        dashboard = json.load(f)

        sample_r = []
        for i in dashboard['projections']:
            sample_r.append(dashboard['projections'][i]['sample_r'])
        sample_r = np.array(sample_r)
        # here we assume that we always take only one 360 rotation. I don't know what happens if you do multiple
        angles = ((sample_r/360.0)*2*np.pi).astype(np.float32)
        f.close()
        return angles
    else:
        return None


def get_detector_offsets(jsonfile=''):
    if os.path.exists(jsonfile):
        f = open(jsonfile)
        dashboard = json.load(f)

        detector_x = []
        detector_y = []
        for i in dashboard['projections']:
            detector_x.append(dashboard['projections'][i]['detector_x'])
            detector_y.append(dashboard['projections'][i]['detector_y'])
        detector_x = np.array(detector_x)
        detector_y = np.array(detector_y)
        # detector offsetts are calculated with respect to the first projection, assuming that it is the proposed central scan setup.
        detector_x_offsets = -1*(detector_x-detector_x[0])
        detector_y_offsets = -1*(detector_y-detector_y[0])
        f.close()
        return detector_x_offsets, detector_y_offsets
    else:
        return None


def get_samplestage_z_offset(jsonfile=''):
    offset = 0.0
    if os.path.exists(jsonfile):
        f = open(jsonfile)
        dashboard = json.load(f)
        # for now we assume that the sample stage positions are constant for the entire scan
        offset = dashboard['projections']['0000']['sample_z']
        f.close()

    return offset


def get_exposure_time_projection(jsonfile=''):
    exptimetotal = 0.0
    if os.path.exists(jsonfile):
        f = open(jsonfile)
        dashboard = json.load(f)
        ntriggers = dashboard['acquisition']['nr_of_images']
        exptime = dashboard['acquisition']['exposure_time']
        exptimetotal = ntriggers*exptime
        f.close()

    return exptimetotal

# this can possibly be done much faster with parallel processing


def get_dac_settings(jsonfile=''):
    th0_dacs = []
    th1_dacs = []
    if os.path.exists(jsonfile):
        f = open(jsonfile)
        dashboard = json.load(f)
        th0_dacs.append(dashboard['acquisition']['th0_0'])
        th0_dacs.append(dashboard['acquisition']['th0_1'])
        th0_dacs.append(dashboard['acquisition']['th0_2'])
        th0_dacs.append(dashboard['acquisition']['th0_3'])
        th1_dacs.append(dashboard['acquisition']['th1_0'])
        th1_dacs.append(dashboard['acquisition']['th1_1'])
        th1_dacs.append(dashboard['acquisition']['th1_2'])
        th1_dacs.append(dashboard['acquisition']['th1_3'])
        f.close()

    return th0_dacs, th1_dacs



def median_filter_projection_set(projections, kernelsize=5):
    return np.array(Parallel(n_jobs=60)(delayed(median_filter_2D)(x, kernelsize) for x in projections))


def median_filter_2D(projection, kernelsize=5):
    return medfilt2d(projection, kernelsize)


def apply_badmap_to_projections(projections, badmap):
    # a = list()
    # for x in projections:
    #     a.append(apply_badmap_to_projection(x, badmap))
    # return np.array(a)
    return np.array(Parallel(n_jobs=60)(delayed(apply_badmap_to_projection)(x, badmap) for x in projections))


def apply_badmap_to_projection(projection, badmap):
    assert isinstance(projection, np.ndarray)
    assert isinstance(badmap, np.ndarray)
    assert projection.shape == badmap.shape
    assert projection.shape[0] == projection.shape[1]
    assert projection.shape[0] % 256 == 0

    lproj = projection * badmap
    valid_mask = (lproj > 0)
    coords = np.array(np.nonzero(valid_mask)).T
    values = lproj[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
    return it(list(np.ndindex(lproj.shape))).reshape(lproj.shape)


class ConeGeometryJasper2(Geometry):
    ''' Some of these parameters are overwritten with better values in the recon_scan functions '''

    def __init__(self, gReconParams: dict, high_quality: bool = True, nVoxel=None):

        Geometry.__init__(self)
        pixels = gReconParams['pixels']
        pixel_pitch_mm = gReconParams['pixel_pitch']
        DSD = gReconParams['distance_source_detector']
        DOD = gReconParams['distance_object_detector']
        rotDetector = gReconParams['detector_rotation']

        total_detector_size_mm = pixels * pixel_pitch_mm  # 512 * 0.055 mm = 28.16 mm

        # VARIABLE                                                DESCRIPTION                   UNITS
        # ------------------------------------------------------------------------------------------------
        if high_quality:
            self.nDetector = np.array((pixels, pixels))         # number of pixels              (px)

            ''' nVoxel and sVoxel are overwritten later... '''
            self.nVoxel = np.array((pixels, pixels, pixels))    # number of voxels              (vx)
            self.sVoxel = np.array((total_detector_size_mm,
                                    total_detector_size_mm,
                                    total_detector_size_mm))    # total size of the image       (mm)
        else:
            number_of_voxels = 1
            number_of_pixels = 11

            ''' nVoxel and sVoxel are overwritten later... '''
            self.nDetector = np.array((number_of_pixels,
                                       pixels))                # number of pixels              (px)
            self.nVoxel = np.array((number_of_voxels,
                                    pixels,
                                    pixels))                    # number of voxels              (vx)
            self.sVoxel = np.array((pixel_pitch_mm*number_of_voxels,
                                    pixels*pixel_pitch_mm,
                                    pixels*pixel_pitch_mm))    # total size of the image       (mm)

        ''' We will set the common variables last because other variables depend on some of them '''
        # VARIABLE                                                DESCRIPTION                   UNITS
        # ------------------------------------------------------------------------------------------------
        self.DSD = DSD                                          # Distance Source Detector      (mm)
        self.DSO = DSD - DOD                                    # Distance Source Origin        (mm)
        # Detector parameters
        # *2 because of the pixel geometry definition probably
        self.dDetector = np.array((pixel_pitch_mm,
                                   pixel_pitch_mm))           # size of each pixel            (mm)
        self.sDetector = self.nDetector * self.dDetector        # total size of the detector    (mm)

        # Offsets
        # Offset of image from origin   (mm)
        self.offOrigin = np.array((0, 0, 0))
        self.offDetector = np.array((0, 0))                     # Offset of Detector            (mm)
        # Detector rotation             (radians)
        self.rotDetector = rotDetector
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
            # Accuracy of FWD proj          (vx/sample)
            self.accuracy = 0.5
            # Mode
            self.mode = 'cone'                                  # parallel, cone'''


def find_optimal_offset(gReconParams: dict, projections, angles, detector_x_offsets, detector_y_offsets, stage_offset=0.0, search_range=70):
    ''' TODO Replace this! It does not work

    This function takes quite a while, which can probably be optimized.
    On the other hand, we only have to do this once per scan setup, and then store the value for later use.
    '''

    projection0 = median_filter(projections[0, :, :], 5)
    projection180 = median_filter(np.flip(projections[round(projections.shape[0]/2)-1, :, :], 1), 5)
    print(detector_x_offsets[0], detector_x_offsets[round(projections.shape[0]/2)-1])
    print(detector_y_offsets[0], detector_y_offsets[round(projections.shape[0]/2)-1])

    shift, error, diffphase = phase_cross_correlation(projection0, projection180,
                                                      upsample_factor=100)

    estimate = -shift.item(1)/2
    print(estimate)

    #########################################
   # estimate = 0.47/gReconParams['pixel_pitch']  # px
    ##########################################
    if stage_offset != 0.0:
        estimate = stage_offset/gReconParams['pixel_pitch']  # px
    geom = ConeGeometryJasper2(gReconParams, high_quality=False)
    # we should avoid pixels from the cross of the detector
    yshift = 3
    # we also need to take into account that detector motion up to 5 pixels could have been used during the acquisition
    projections_small = projections[:, int(
        projections.shape[1]/2)-6-yshift:int(projections.shape[1]/2)+5-yshift, :]
    projections_small = median_filter_projection_set(projections_small)
    max_value = -9999
    opt_shift = -99
    cnt = 0
    xvalues = []
    yvalues = []

    for i in trange(int(estimate*10) - search_range, int(estimate*10) + search_range, 1):
        # note that the detector x and y axis are opposite the x and y coordinate system of the geometry
        geom.offDetector = np.vstack(
            (detector_y_offsets, detector_x_offsets+(i/10) * gReconParams['pixel_pitch'])).T
        geom.rotDetector = np.array(gReconParams['detector_rotation'])
        geom.DSD = gReconParams['distance_source_detector']
        # stageoffset = 0
        geom.DSO = geom.DSD - gReconParams['distance_object_detector']  # - stageoffset

        imgfdk = algs.fdk(projections_small, geom, angles)
        im = imgfdk[0, :, :].astype(np.float32)
        dft = cv2.dft(im, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        value = np.mean(magnitude_spectrum)
        if value > max_value:
            max_value = value
            opt_shift = i
        xvalues.append((i/10) * gReconParams['pixel_pitch'])
        yvalues.append(value)

    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.scatter(xvalues, yvalues)
    plt.show()
    return (opt_shift/10) * gReconParams['pixel_pitch']


def make_Nifti1Image(array, voxel_size: float):
    img = nib.Nifti1Image(array, np.eye(4))
    img.header['pixdim'][1:4] = voxel_size
    return img


def recon_scan(gReconParams: dict, projections, angles, detector_x_offsets, detector_y_offsets, centre_of_rotation_offset_x_mm, centre_of_rotation_offset_y_mm, high_quality=True):
    # sample_z_offset --> 0 - 100 mm

    geometry = ConeGeometryJasper2(gReconParams, high_quality=high_quality)
    geometry.offDetector = np.vstack(
        (detector_y_offsets + centre_of_rotation_offset_y_mm, detector_x_offsets + centre_of_rotation_offset_x_mm)).T
    geometry.rotDetector = np.array(gReconParams['detector_rotation'])
    geometry.DSD = gReconParams['distance_source_detector']
    geometry.DSO = geometry.DSD - gReconParams['distance_object_detector']

    print('dsd: ', geometry.DSD, 'dso: ', geometry.DSO)

    # number of voxels              (vx)
    geometry.nVoxel = np.array(gReconParams['recon_voxels'])
    # total size of the image       (mm)
    geometry.sVoxel = np.array(gReconParams['recon_size'])
    # size of each voxel            (mm)
    geometry.dVoxel = geometry.sVoxel/geometry.nVoxel
    geometry.offOrigin = np.array((0, 0, 0))

    imgfdk = algs.fdk(projections, geometry, angles, filter='cosine')
    imgfdk[imgfdk < 0] = 0
    img = (imgfdk*10000).astype(np.int32)

    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.flip(img, 2)

    return img, geometry


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


def load_or_generate_data_arrays(base_json_file, base_folder, results_folder, gReconParams):
    exp_time = []
    th0_list = []
    th1_list = []
    th0_dac_list = []
    th1_dac_list = []
    spectral_projs_th0 = []
    spectral_open_th0 = []
    spectral_projs_th1 = []
    spectral_open_th1 = []
    numpy_output_files = ['projs_stack_th0.npy', 'open_stack_th0.npy',
                          'projs_stack_th1.npy', 'open_stack_th1.npy', 'thlist_th0.npy', 'thlist_th1.npy']

    if os.path.exists(base_json_file):
        f = open(base_json_file)
        dashboard = json.load(f)

        if files_exist(results_folder, numpy_output_files):
            print('Loading existing numpy files, should take <7.5 seconds')

            spectral_projs_th0 = np.load(os.path.join(results_folder, numpy_output_files[0]))
            spectral_open_th0 = np.load(os.path.join(results_folder, numpy_output_files[1]))
            spectral_projs_th1 = np.load(os.path.join(results_folder, numpy_output_files[2]))
            spectral_open_th1 = np.load(os.path.join(results_folder, numpy_output_files[3]))
            th0_list = np.load(os.path.join(results_folder, numpy_output_files[4]))
            th1_list = np.load(os.path.join(results_folder, numpy_output_files[5]))

            assert spectral_projs_th0.size > 1  # 1 byte minimum
            assert spectral_open_th0.size > 1  # 1 byte minimum
            assert spectral_projs_th1.size > 1  # 1 byte minimum
            assert spectral_open_th1.size > 1  # 1 byte minimum
            assert th0_list.size > 1  # 1 byte minimum
            assert th1_list.size > 1  # 1 byte minimum

            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                open_image_json = scan_json
                folder_string = dashboard['thresholdscan'][i]['projectionsfolder']

                th0_keV = folder_string[0:folder_string.find('_')]
                th1_keV = folder_string[folder_string.find('_')+1:]
                th0_dacs, th1_dacs = get_dac_settings(scan_json)
                th0_dac_list.append(th0_dacs)
                th1_dac_list.append(th1_dacs)
                exp_time.append(get_exposure_time_projection(scan_json))
            exp_time = np.asarray(exp_time)
            th0_dac_list = np.asarray(th0_dac_list)
            th1_dac_list = np.asarray(th1_dac_list)

        else:
            print(f'Making new numpy files, should take ~4.5 minutes. At least one file was missing :( ')

            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                if 'openimagesfolder' in dashboard:
                    open_image_folder = os.path.join(
                        base_folder, dashboard['thresholdscan'][i]['openimagesfolder'])
                else:
                    open_image_folder = scan_folder
                if 'openimages_json' in dashboard:
                    open_image_json = os.path.join(
                        open_image_folder, dashboard['thresholdscan'][i]['openimages_json'])
                else:
                    open_image_json = scan_json

                folder_string = dashboard['thresholdscan'][i]['projectionsfolder']
                th0_keV = folder_string[0:folder_string.find('_')]
                th1_keV = folder_string[folder_string.find('_')+1:]
                exp_time.append(get_exposure_time_projection(scan_json))

                th0_list.append(float(th0_keV))
                th1_list.append(float(th1_keV))
                th0_keV = folder_string[0:folder_string.find('_')]
                th1_keV = folder_string[folder_string.find('_')+1:]

                th0_dacs, th1_dacs = get_dac_settings(scan_json)
                th0_dac_list.append(th0_dacs)
                th1_dac_list.append(th1_dacs)
                projs_th0 = projectionsloader(
                    scan_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                openimg_th0 = openimgloader(
                    open_image_json, th0=True, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                projs_th1 = projectionsloader(
                    scan_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                openimg_th1 = openimgloader(
                    open_image_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                spectral_projs_th0.append(projs_th0)
                spectral_open_th0.append(openimg_th0)
                spectral_projs_th1.append(projs_th1)
                spectral_open_th1.append(openimg_th1)

            spectral_projs_th0 = np.asarray(spectral_projs_th0)
            spectral_open_th0 = np.asarray(spectral_open_th0)
            spectral_projs_th1 = np.asarray(spectral_projs_th1)
            spectral_open_th1 = np.asarray(spectral_open_th1)
            exp_time = np.asarray(exp_time)
            th0_list = np.asarray(th0_list)
            th1_list = np.asarray(th1_list)
            th0_dac_list = np.asarray(th0_dac_list)
            th1_dac_list = np.asarray(th1_dac_list)
            np.save(os.path.join(results_folder, 'projs_stack_th0.npy'), spectral_projs_th0)
            np.save(os.path.join(results_folder, 'open_stack_th0.npy'), spectral_open_th0)
            np.save(os.path.join(results_folder, 'projs_stack_th1.npy'), spectral_projs_th1)
            np.save(os.path.join(results_folder, 'open_stack_th1.npy'), spectral_open_th1)
            np.save(os.path.join(results_folder, 'thlist_th0.npy'), th0_list)
            np.save(os.path.join(results_folder, 'thlist_th1.npy'), th1_list)

        angles = get_proj_angles(scan_json)
        z_offset = get_samplestage_z_offset(scan_json)
        detector_x_offsets, detector_y_offsets = get_detector_offsets(scan_json)

        return spectral_projs_th0, spectral_open_th0, spectral_projs_th1, spectral_open_th1, th0_list, th1_list, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets, th0_dac_list, th1_dac_list
    else:
        return None


def files_exist(path: str, file_list: List[str]) -> bool:
    ''' If a file does not exist or is less than the threshold, return False '''
    for f in file_list:
        full_path = os.path.join(path, f)
        if not os.path.exists(full_path):
            return False
    return True


def save_and_or_load_npy_files(path: str, array_filename: str, generating_function):
    p = os.path.join(path, array_filename)
    if os.path.exists(p):
        print(f'Loading existing numpy file: {p}')
        my_array = np.load(p)
    else:
        my_array = generating_function()
        print(f'Saving numpy file: {p}')
        np.save(p, my_array)
    return my_array


def calculate_mean_of_non_cross_pixels(gReconParams: dict, arr: np.ndarray, idx: int):
    # e.g. arr --> 12, 512, 512
    # idx 0 --> 0-12

    a0 = 0
    a1 = (gReconParams['pixels'] // 2) - 1  # 255 for 512 pixels
    a2 = (gReconParams['pixels'] // 2) + 1  # 257 for 512 pixels
    a3 = gReconParams['pixels']
    mean = 0.25 * (
        np.mean(arr[idx, a0:a1, a0:a1])
        + np.mean(arr[idx, a0:a1, a2:a3])
        + np.mean(arr[idx, a2:a3, a0:a1])
        + np.mean(arr[idx, a2:a3, a2:a3])
    )

    return mean


def generate_correct_dac_values(gReconParams: dict, open_mean_all_thr: np.ndarray, dac_list_all_chips: np.ndarray, plot: bool = False, poly_order = 2, open_img_path = None) -> np.ndarray:

    def power(arr: np.ndarray, pow: int) -> np.ndarray:
        return np.array([x**pow for x in arr])
    
    def solve_poly_for_x(poly_coeffs: np.ndarray, y: float) -> np.ndarray:
        pc = poly_coeffs.copy()
        pc[-1] -= y
        return np.roots(pc)
    
    def get_chip_dac_from_array_using_orientation(dac_per_chip_list: np.ndarray, i: int, j: int, half_pixels: int, chip_indices: list) -> np.ndarray:
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
    assert dac_list_all_chips.shape[0] >= 8 # N thresholds (DAC units) scanned over / acquired simultaneously - at least 8 should have been measured...
    assert dac_list_all_chips.shape[1] == 4 # Always 4 chips
    

    half_pixels = gReconParams['pixels'] // 2
    dac_correct = np.zeros((len(dac_list_all_chips), gReconParams['pixels'], gReconParams['pixels']))
    open_img_out = np.zeros((len(dac_list_all_chips), gReconParams['pixels'], gReconParams['pixels']), dtype=np.float32)
    chip_indices = (0, 1, 2, 3) # TL, TR, BR, BL    Dexter = (0, 1, 2, 3)    Serval UP orientation = (2, 3, 0, 1)

    a0 = 0
    a1 = (gReconParams['pixels'] // 2) - 1 # 255 for 512 pixels
    a2 = (gReconParams['pixels'] // 2) + 1 # 257 for 512 pixels
    a3 = gReconParams['pixels']
    mean = np.zeros((len(dac_list_all_chips)))

    for dac in trange(0,len(dac_list_all_chips)): # assuming that dac_list_all_chips is just for either th0 or th1
        mean[dac] = 0.25 * (
            np.mean(open_mean_all_thr[dac, a0:a1, a0:a1]) +
            np.mean(open_mean_all_thr[dac, a0:a1, a2:a3]) +
            np.mean(open_mean_all_thr[dac, a2:a3, a0:a1]) +
            np.mean(open_mean_all_thr[dac, a2:a3, a2:a3])
        )   
        # print(f'DAC = {dac_list_all_chips[dac]}, mean = {mean[dac]}')

    for i in trange(0, open_mean_all_thr.shape[1]):
        for j in range(0, open_mean_all_thr.shape[2]):
            x_dacs_of_this_chip = get_chip_dac_from_array_using_orientation(dac_list_all_chips, i, j, half_pixels, chip_indices)
            y_counts = open_mean_all_thr[:, i, j]
            assert x_dacs_of_this_chip.ndim == 1, (x_dacs_of_this_chip.ndim, x_dacs_of_this_chip)
            assert len(x_dacs_of_this_chip) == len(y_counts), (len(x_dacs_of_this_chip), len(y_counts))

            if poly_order == 0:
                raise NotImplementedError("To be implemented")
                
            if poly_order == 1:
                poly_coeffs, _, _, _, _ = np.polyfit(x_dacs_of_this_chip, np.log(y_counts), poly_order, w=np.sqrt(x_dacs_of_this_chip), full=True)
                for dac in range(0, len(dac_list_all_chips)):
                    poly_roots = solve_poly_for_x(poly_coeffs, np.log(mean[dac]))
                    if poly_roots.shape[0] == 0:
                        dac_corr = np.NaN # x_chip_dac[dac]
                    else:
                        dac_corr = poly_roots[0]
                    if (dac_corr > (x_dacs_of_this_chip[dac]*2)) or (dac_corr < (x_dacs_of_this_chip[dac]/2)):
                        dac_corr = np.NaN # x_chip_dac[dac]

                    dac_correct[dac, i, j] = dac_corr
                ''' TODO are the order of these coefficients correct??? '''
                open_img_out[:, i, j] = np.exp(poly_coeffs[1]) * np.exp(poly_coeffs[0]*dac_corr)
            
            elif poly_order == 2:
                poly_coeffs, _, _, _, _ = np.polyfit(x_dacs_of_this_chip, y_counts, poly_order, full=True)
                for dac in range(0, len(dac_list_all_chips)):
                    poly_roots = solve_poly_for_x(poly_coeffs, mean[dac])
                    if poly_roots.shape[0] == 0:
                        dac_corr = np.NaN # x_chip_dac[dac]
                    else:
                        dac_corr = poly_roots[1]
                    if (dac_corr > (x_dacs_of_this_chip[dac]*2)) or (dac_corr < (x_dacs_of_this_chip[dac]/2)):
                        dac_corr = np.NaN # x_chip_dac[dac]

                    dac_correct[dac, i, j] = dac_corr
                open_img_out[:, i, j] = (poly_coeffs[0] * dac_corr**2) + (poly_coeffs[1]*dac_corr) + poly_coeffs[2]
            
            else:
                raise NotImplementedError("To be implemented...?")

            if plot and i < 1 and j < 1:
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
                    
                    plt.errorbar(x_dacs_of_this_chip, y_counts, np.sqrt(y_counts), fmt='', label='Raw counts')
                    plt.plot(dac_correct[:, i, j], mean, marker='+', markersize=10) # zero is added for the frist dac to be displayed
                    plt.plot(fit_x, y_poly_fit, label='fit - poly order 2')

                    plt.legend()
                    plt.show()

    if open_img_path != None:
        for dac in tqdm(dac_list_all_chips):
            sitk.WriteImage(sitk.GetImageFromArray(open_img_out), os.path.join(open_img_path, f'corrected_open_img_{dac}.nii'))

    return dac_correct

    # for i in trange(0, open_mean_all_thr.shape[1]):
    #     for j in range(0, open_mean_all_thr.shape[2]):
    #         x_dacs_of_this_chip = get_chip_dac_from_array_using_orientation(dac_list_all_chips, i, j, half_pixels, chip_indices)
    #         y_counts = open_mean_all_thr[:, i, j]
    #         assert x_dacs_of_this_chip.ndim == 1, (x_dacs_of_this_chip.ndim, x_dacs_of_this_chip)
    #         assert len(x_dacs_of_this_chip) == len(y_counts), (len(x_dacs_of_this_chip), len(y_counts))

    #         if poly_order == 0:
    #             raise NotImplementedError("To be implemented")

    #         if poly_order == 1:
    #             poly_coeffs, _, _, _, _ = np.polyfit(x_dacs_of_this_chip, np.log(y_counts), poly_order, w=np.sqrt(x_dacs_of_this_chip), full=True)
    #             poly_roots = solve_poly_for_x(poly_coeffs, np.log(mean_of_non_cross_pixels))
    #             if poly_roots.shape[0] == 0:
    #                 dac_corr = np.NaN # x_chip_dac[dac_index]
    #             else:
    #                 dac_corr = poly_roots[0]
                
    #             ''' TODO are the order of these coefficients correct??? '''
    #             open_img_out[:, i, j] = np.exp(poly_coeffs[1]) * np.exp(poly_coeffs[0]*dac_corr)

    #         elif poly_order == 2:
    #             poly_coeffs, _, _, _, _ = np.polyfit(x_dacs_of_this_chip, y_counts, poly_order, full=True)
    #             poly_roots = solve_poly_for_x(poly_coeffs, mean_of_non_cross_pixels)
    #             if poly_roots.shape[0] == 0:
    #                 dac_corr = np.NaN # x_chip_dac[dac_index]
    #             else:
    #                 dac_corr = poly_roots[1]
           
    #             open_img_out[:, i, j] = (poly_coeffs[0] * dac_corr**2) + (poly_coeffs[1]*dac_corr) + poly_coeffs[2]
    #         else:
    #             raise NotImplementedError("To be implemented...?")
            
    #         if plot and i < 1 and j < 1:
    #             if poly_order == 0:
    #                 raise NotImplementedError
    #             elif poly_order == 1:
    #                 raise NotImplementedError
    #             elif poly_order == 2:
    #                 # Supersample so we do not get artifacts - no jagged lines, only smooth lines :)
    #                 fit_x = np.linspace(x_dacs_of_this_chip[0], x_dacs_of_this_chip[-1], len(x_dacs_of_this_chip))
    #                 y_poly_fit = (poly_coeffs[0]*power(fit_x, 2)) + \
    #                     (poly_coeffs[1]*power(fit_x, 1)) + poly_coeffs[2]
    #                 plt.xlabel('Threshold (DAC)')
    #                 plt.ylabel('Counts ()')
    #                 plt.title(f'Pixel = {i}, {j}')
                    
    #                 plt.errorbar(x_dacs_of_this_chip, y_counts, np.sqrt(y_counts), fmt='', label='Raw counts')
    #                 plt.plot(dac_corr, mean_of_non_cross_pixels, marker='+', markersize=10)
    #                 plt.plot(fit_x, y_poly_fit, label='fit - poly order 2')

    #                 plt.legend()
    #                 plt.show()
            
    #         ''' If the calculated DAC value is outside the reasonable range,
    #         we pass it as a NaN and correct for it later...
    #         '''

    #         if (dac_corr > (x_dacs_of_this_chip[dac_index]*2)) or (dac_corr < (x_dacs_of_this_chip[dac_index]/2)):
    #             dac_corr = np.NaN # x_chip_dac[dac_index]

    #         dac_correct[i, j] = dac_corr

    # if open_img_path != None:
    #     sitk.WriteImage(sitk.GetImageFromArray(open_img_out), os.path.join(open_img_path, 'corrected_open_img_'+str(dac_index)+'.nii'))

    # return dac_correct


def fit_proj_data_values(data_set: np.ndarray, dacs_list: np.ndarray, polyorder = 2):

    assert len(data_set.shape) == 4  # DAC, proj, x, y
    fit_array = data_set.reshape(data_set.shape[0], -1)

    if polyorder == 1:
        regressions, res, _, _, _ = np.polyfit(dacs_list, np.log(fit_array), polyorder, w=np.sqrt(dacs_list), full=True)
    elif polyorder == 2:
        regressions, res, _, _, _ = np.polyfit(dacs_list, fit_array, polyorder, full=True)

    regressions = regressions.reshape(polyorder+1, data_set.shape[1], data_set.shape[2], data_set.shape[3])
    residuals = res.reshape(data_set.shape[1], data_set.shape[2], data_set.shape[3])
    return regressions, residuals


def generate_bad_pixel_corrected_array(ofc, gReconParams):
    print('Doing median filter on OFC data...')
    ofc_mf = median_filter_projection_set(ofc, 5)
    diff_mf = np.abs(ofc-ofc_mf)
    meanmap = np.mean(diff_mf, axis=0)
    stdmap = np.std(diff_mf, axis=0)
    badmap = np.ones((gReconParams['pixels'], gReconParams['pixels']))
    half = np.int32(badmap.shape[0]/2)
    badmap[half-2:half+1] = 0
    badmap[:, half-2:half+1] = 0

    ''' TODO These might be too aggressive, review... '''
    badmap[meanmap > 0.2] = 0
    badmap[stdmap > 0.05] = 0
    ofc_bpc = apply_badmap_to_projections(ofc, badmap)
    return ofc_bpc


if __name__ == "__main__":
    freeze_support()
    print('You executed the wrong file LOL!!!! ')
