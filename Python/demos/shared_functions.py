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
        if fillgap:
            gap = 2
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
            cross[256:258, 256:258] = np.nan
            finalarray = cross[1:-1, 1:-1]

        if badpixelcorr:
            finalarray[finalarray == np.inf] = 0
            finalarray[finalarray == -np.inf] = 0
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


def median_filter_projection_set(projections, kernelsize=5):
    with multiprocessing.Pool(60) as pool:
        processes = [pool.apply_async(median_filter_2D, args=(x, kernelsize)) for x in projections]
        result = [p.get() for p in processes]
        return np.asarray(result)

    # lprojs = np.zeros_like(projections)
    # for i in range(0, projections.shape[0]):
    #     lprojs[i, :, :] = medfilt2d(projections[i, :, :], kernelsize)
    # return (lprojs)


def median_filter_2D(projection, kernelsize=5):
    return medfilt2d(projection, kernelsize)


def apply_badmap_to_projections(projections, badmap):
    with multiprocessing.Pool(60) as pool:
        processes = [pool.apply_async(apply_badmap_to_projection, args=(x, badmap))
                     for x in projections]
        # for p in processes:
        # print(p.get())
        result = [p.get() for p in processes]
        return np.asarray(result)


def apply_badmap_to_projection(projection, badmap):
    lproj = projection * badmap
    valid_mask = (lproj > 0)
    coords = np.array(np.nonzero(valid_mask)).T
    values = lproj[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
    return it(list(np.ndindex(lproj.shape))).reshape(lproj.shape)


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
            self.accuracy = 0.5                                 # Accuracy of FWD proj          (vx/sample)
            # Mode
            self.mode = 'cone'                                  # parallel, cone'''


def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)


# def find_optimal_offset(gReconParams: dict, projections, angles, detector_x_offsets, detector_y_offsets, stage_offset=0, search_range=70):
#     ''' TODO Replace this! It does not work

#     This function takes quite a while, which can probably be optimized.
#     On the other hand, we only have to do this once per scan setup, and then store the value for later use.
#     '''

#     projection0 = projections[0, :, :]
#     projection180 = np.flip(projections[round(projections.shape[0]/2)-1, :, :], 1)

#     shift, error, diffphase = phase_cross_correlation(projection0, projection180,
#                                                       upsample_factor=100)

#     estimate = -shift.item(1)/2
#     print(estimate)
#     #########################################
#     # estimate = 4.9 # px
#     ##########################################

#     geom = ConeGeometryJasper(gReconParams, high_quality=False)
#     # we should avoid pixels from the cross of the detector
#     yshift = 3
#     # we also need to take into account that detector motion up to 5 pixels could have been used during the acquisition
#     projections_small = projections[:, int(
#         projections.shape[1]/2)-6-yshift:int(projections.shape[1]/2)+5-yshift, :]

#     max_value = -9999
#     opt_shift = -99
#     cnt = 0
#     xvalues = []
#     yvalues = []

#     for i in trange(int(estimate*100) - search_range, int(estimate*100) + search_range, 1):
#         # note that the detector x and y axis are opposite the x and y coordinate system of the geometry
#         geom.offDetector = np.vstack((detector_y_offsets, detector_x_offsets+(i/100) * gReconParams['pixel_pitch'])).T
#         geom.rotDetector = np.array(gReconParams['detector_rotation'])
#         geom.DSD = gReconParams['distance_source_detector']
#         # stageoffset = 0
#         geom.DSO = geom.DSD - gReconParams['distance_object_detector'] # - stageoffset

#         imgfdk = algs.fdk(projections_small, geom, angles)
#         im = imgfdk[0, :, :].astype(np.float32)
#         dft = cv2.dft(im, flags=cv2.DFT_COMPLEX_OUTPUT)
#         dft_shift = np.fft.fftshift(dft)

#         magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
#         value = np.mean(magnitude_spectrum)
#         if value > max_value:
#             max_value = value
#             opt_shift = i
#         xvalues.append((i/100) * gReconParams['pixel_pitch'])
#         yvalues.append(value)

#     # plt.xlabel('X values')
#     # plt.ylabel('Y values')
#     # plt.scatter(xvalues, yvalues)
#     # plt.show()
#     return (opt_shift/100) * gReconParams['pixel_pitch']


def recon_scan(gReconParams: dict, projections, angles, sample_z_offset, detector_x_offsets, detector_y_offsets, global_detector_shift_y, high_quality=True):
    # sample_z_offset --> 0 - 100 mm

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

    imgfdk = algs.fdk(projections, geo, angles, filter='cosine')
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


def load_or_generate_data_arrays(base_json_file, base_folder, results_folder):
    if os.path.exists(base_json_file):
        f = open(base_json_file)
        dashboard = json.load(f)
        exp_time = []
        # th0_list = []
        # th1_list = []

        numpy_output_files = ['projs_stack_th0.npy', 'open_stack_th0.npy']
        if files_exist(results_folder, numpy_output_files):
            print('Loading existing numpy files, should take <7.5 seconds')

            spectral_projs_th0 = np.load(os.path.join(results_folder, numpy_output_files[0]))
            spectral_open_th0 = np.load(os.path.join(results_folder, numpy_output_files[1]))
            # spectral_projs_th1 = np.load(os.path.join(results_folder, numpy_output_files[2]))
            # spectral_open_th1 = np.load(os.path.join(results_folder, numpy_output_files[3]))
            # th0_list = np.load(os.path.join(results_folder, numpy_output_files[4]))
            # th1_list = np.load(os.path.join(results_folder, numpy_output_files[5]))

            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                open_image_json = scan_json
                # folder_string = dashboard['thresholdscan'][i]['projectionsfolder']

                # th0_keV = folder_string[0:folder_string.find('_')]
                # th1_keV = folder_string[folder_string.find('_')+1:]

                exp_time.append(get_exposure_time_projection(scan_json))
            exp_time = np.asarray(exp_time)

        else:
            print(f'Making new numpy files, should take ~4.5 minutes. At least one file was missing :( ')

            spectral_projs_th0 = []
            spectral_open_th0 = []
            # spectral_projs_th1 = []
            # spectral_open_th1 = []
            # th0_list = []
            # th1_list = []
            for i in tqdm(dashboard['thresholdscan']):
                scan_folder = os.path.join(
                    base_folder, dashboard['thresholdscan'][i]['projectionsfolder'])
                scan_json = os.path.join(
                    scan_folder, dashboard['thresholdscan'][i]['projections_json'])
                open_image_json = scan_json
                # folder_string = dashboard['thresholdscan'][i]['projectionsfolder']
                # th0_keV = folder_string[0:folder_string.find('_')]
                # th1_keV = folder_string[folder_string.find('_')+1:]
                exp_time.append(get_exposure_time_projection(scan_json))
                # th0_list.append(float(th0_keV))
                # th1_list.append(float(th1_keV))

                projs_th0 = projectionsloader(
                    scan_json, th0=True, badpixelcorr=False, medianfilter=False, fillgap=False)
                openimg_th0 = openimgloader(
                    open_image_json, th0=True, badpixelcorr=False, medianfilter=False, fillgap=False)
                # projs_th1 = projectionsloader(scan_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                # openimg_th1 = openimgloader(open_image_json, th0=False, badpixelcorr=gReconParams['bad_pixel_correction'], medianfilter=gReconParams['median_filter'], fillgap=gReconParams['fill_gap'])
                spectral_projs_th0.append(projs_th0)
                spectral_open_th0.append(openimg_th0)
                # spectral_projs_th1.append(projs_th1)
                # spectral_open_th1.append(openimg_th1)

            spectral_projs_th0 = np.asarray(spectral_projs_th0)
            spectral_open_th0 = np.asarray(spectral_open_th0)
            # spectral_projs_th1 = np.asarray(spectral_projs_th1)
            # spectral_open_th1 = np.asarray(spectral_open_th1)
            exp_time = np.asarray(exp_time)
            # th0_list = np.asarray(th0_list)
            # th1_list = np.asarray(th1_list)

            np.save(os.path.join(results_folder, 'projs_stack_th0.npy'), spectral_projs_th0)
            np.save(os.path.join(results_folder, 'open_stack_th0.npy'), spectral_open_th0)
            # np.save(os.path.join(results_folder, 'projs_stack_th1.npy'), spectral_projs_th1)
            # np.save(os.path.join(results_folder, 'open_stack_th1.npy'), spectral_open_th1)
            # np.save(os.path.join(results_folder, 'thlist_th0.npy'), th0_list)
            # np.save(os.path.join(results_folder, 'thlist_th1.npy'), th1_list)

    angles = get_proj_angles(scan_json)
    z_offset = get_samplestage_z_offset(scan_json)
    detector_x_offsets, detector_y_offsets = get_detector_offsets(scan_json)

    return spectral_projs_th0, spectral_open_th0, exp_time, angles, z_offset, detector_x_offsets, detector_y_offsets


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


def generate_dac_values(gReconParams, open_mean_th0, th0_list, mean, th):
    DAC_values = np.zeros((gReconParams['pixels'], gReconParams['pixels']))
    regression_out = np.zeros((3, gReconParams['pixels'], gReconParams['pixels']))
    for i in trange(0, open_mean_th0.shape[1]):
        for j in range(0, open_mean_th0.shape[2]):
            yvalues = open_mean_th0[:, i, j]
            regressions, _, _, _, _ = np.polyfit(th0_list, yvalues, 2, full=True)
            regression_out[:, i, j] = regressions
            DAC = solve_for_y(regressions, mean)[1]
            if (DAC > th0_list[th]*2) or (DAC < th0_list[th]/2):
                DAC = th0_list[th]
            DAC_values[i, j] = DAC
    return DAC_values


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
    badmap[meanmap > 0.2] = 0
    badmap[stdmap > 0.05] = 0
    ofc_bpc = apply_badmap_to_projections(ofc, badmap)
    return ofc_bpc


if __name__ == "__main__":
    freeze_support()
    print('You executed the wrong file LOL!!!! ')
