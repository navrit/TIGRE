import os
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.ndimage import median_filter
import json
import multiprocessing
from scipy.signal import medfilt2d


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
        pool = multiprocessing.Pool(60)
        processes = [pool.apply_async(load_projection, args=(
            x, badpixelcorr, medianfilter, fillgap)) for x in totalfilelist]
        result = [p.get() for p in processes]
        return np.asarray(result)
    else:
        return None


def openimgloader(jsonfile='', th0=True, badpixelcorr=True, medianfilter=False, fillgap=False):
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
        pool = multiprocessing.Pool(60)
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
    pool = multiprocessing.Pool(60)
    processes = [pool.apply_async(median_filter_2D, args=(x, kernelsize)) for x in projections]
    result = [p.get() for p in processes]
    return np.asarray(result)

    lprojs = np.zeros_like(projections)
    for i in range(0, projections.shape[0]):
        lprojs[i, :, :] = medfilt2d(projections[i, :, :], kernelsize)
    return (lprojs)


def median_filter_2D(projection, kernelsize=5):
    return medfilt2d(projection, kernelsize)


def apply_badmap_to_projections(projections, badmap):
    pool = multiprocessing.Pool(60)
    processes = [pool.apply_async(apply_badmap_to_projection, args=(x, badmap))
                 for x in projections]
    result = [p.get() for p in processes]
    return np.asarray(result)


def apply_badmap_to_projection(projection, badmap):
    lproj = projection * badmap
    valid_mask = (lproj > 0)
    coords = np.array(np.nonzero(valid_mask)).T
    values = lproj[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
    return it(list(np.ndindex(lproj.shape))).reshape(lproj.shape)
