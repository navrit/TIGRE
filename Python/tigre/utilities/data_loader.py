from __future__ import division
import os
import numpy as np
import scipy.io
from scipy import interpolate
from scipy.ndimage import median_filter, gaussian_filter
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy import ndimage

kernel = Gaussian2DKernel(x_stddev=2)


def load_head_phantom(number_of_voxels=None):
    if number_of_voxels is None:
        number_of_voxels = np.array((128, 128, 128))
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname,'MRheadbrain')
    dirname = os.path.join(dirname,'head.mat')
    test_data = scipy.io.loadmat(dirname)

    # Loads data in F_CONTIGUOUS MODE (column major), convert to Row major
    image = test_data['img'].transpose(2,1,0).copy()
    image_dimensions = image.shape

    zoom_x = number_of_voxels[0] / image_dimensions[0]
    zoom_y = number_of_voxels[1] / image_dimensions[1]
    zoom_z = number_of_voxels[2] / image_dimensions[2]

    # TODO: add test for this is resizing and not simply zooming
    resized_image = scipy.ndimage.interpolation.zoom(image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False)

    return resized_image


def load_cube(number_of_voxels=None):
    if number_of_voxels is None:
        number_of_voxels = np.array((128, 128, 128))

    test_data = scipy.io.loadmat('Test_data/cube.mat')

    # Loads data in F_CONTIGUOUS MODE (column major), convert to Row major
    image = test_data['cube'].copy(order='F')

    image_dimensions = image.shape

    zoom_x = number_of_voxels[0] / image_dimensions[0]
    zoom_y = number_of_voxels[1] / image_dimensions[1]
    zoom_z = number_of_voxels[2] / image_dimensions[2]

    # TODO: add test for this is resizing and not simply zooming
    resized_image = scipy.ndimage.interpolation.zoom(image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False)

    return resized_image

def load_projections(filepath=''):
    if os.path.exists(filepath):
        final = []
        for fname in os.listdir(filepath):
            im = Image.open(os.path.join(filepath, fname))
            imarray = np.array(im)
            dead_pixels = np.isinf(imarray)
            blurred = median_filter(imarray, size=2, mode="mirror")
            filtered = np.ma.where(dead_pixels, blurred, imarray)
#             filtered = np.fliplr(filtered)
#             imarray = np.nan_to_num(imarray, neginf=0) 
            final.append(filtered)

        final = np.asarray(final) # shape = (60000,28,28)
        print(np.max(final), np.min(final))
        return final
    
def load_rayence_projections(filepath='', halvedata=False, flip = True, badpixelcorr = False):
    if os.path.exists(filepath):
        final = []
        cnt = 0
        for fname in tqdm(os.listdir(filepath)):
            if halvedata:
                if cnt % 2 == 1:
                    cnt += 1
                    continue
            cnt += 1    
            filename, file_extension = os.path.splitext(fname)
            if file_extension.lower() == '.raw':
                rawData = open(os.path.join(filepath, fname), 'rb').read()
                imgSize = (1176,1104)
# Use the PIL raw decoder to read the data.
# the 'F;16' informs the raw decoder that we are reading 
# a little endian, unsigned integer 16 bit data.
                im = Image.frombytes('I;16L', imgSize, rawData, 'raw')
            elif file_extension.lower() == '.tif':
                im = Image.open(os.path.join(filepath, fname))
            elif file_extension.lower() == '.tiff':
                im = Image.open(os.path.join(filepath, fname))
            else:
                continue
            imarray = np.array(im)
            if flip:
                imarray = np.flipud(imarray)
                imarray = np.fliplr(imarray)

            if (np.min(imarray)==0) & (badpixelcorr):
                foo = np.array(imarray)
                # Compute the median of the non-zero elements
                m = np.median(foo[foo > 0])
                # Assign the median to the zero elements 
                foo[foo == 0] = m 
                imarray = np.copy(foo)
#                 valid_mask = (imarray > 0)
#                 coords = np.array(np.nonzero(valid_mask)).T
#                 values = imarray[valid_mask]
# #                 it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
#                 it = interpolate.NearestNDInterpolator(coords, values)
#                 imarray = it(list(np.ndindex(imarray.shape))).reshape(imarray.shape)
            final.append(imarray)
    
#             if (np.min(imarray)==0) & (badpixelcorr):
#                 valid_mask = (imarray > 0)
#                 coords = np.array(np.nonzero(valid_mask)).T
#                 values = imarray[valid_mask]
# #                 it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
#                 it = interpolate.NearestNDInterpolator(coords, values)
#                 imarray = it(list(np.ndindex(imarray.shape))).reshape(imarray.shape)
#             final.append(imarray)

        
        final = np.asarray(final) 
#         minimum = np.min(final)
#         final  = final-minimum +1
#         final = -np.log(final)
#         air = np.percentile(final, 5)
#         final = final-air
        return final
    
def load_raw_files(filepath='', flip=True):
    if os.path.exists(filepath):
        final = []
        cnt = 0
        for fname in os.listdir(filepath):
            filename, file_extension = os.path.splitext(fname)
            
            if file_extension.lower() == '.raw':
                rawData = open(os.path.join(filepath, fname), 'rb').read()
                imgSize = (1176,1104)
                im = Image.frombytes('I;16L', imgSize, rawData, 'raw')
            elif file_extension.lower() == '.tif':
                im = Image.open(os.path.join(filepath, fname))
            elif file_extension.lower() == '.tiff':
                im = Image.open(os.path.join(filepath, fname))                
#                 print(os.path.join(filepath, fname))
            else:
                continue
            imarray = np.array(im)
            if flip:
                imarray = np.flipud(imarray)
                imarray = np.fliplr(imarray)
            final.append(imarray)
      
        final = np.asarray(final) 
        return final    

def load_medipix3_files(filepath=''):
    if os.path.exists(filepath):
        final = []
        cnt = 0
        for fname in os.listdir(filepath):
            filename, file_extension = os.path.splitext(fname)
            
            if file_extension.lower() == '.tiff':
                im = Image.open(os.path.join(filepath, fname))                
#                 print(os.path.join(filepath, fname))
            else:
                continue
            imarray = np.array(im)
            imarray = np.flipud(imarray)
            imarray = np.fliplr(imarray)
            print(imarray.shape)
            if imarray.shape[0] == 256:
                gap = 2
            elif imarray.shape[0] == 512:
                gap = 4
            else:
                break
                
            cross = np.zeros((imarray.shape[0]+gap, imarray.shape[1]+gap))
            half = np.int(imarray.shape[0]/2)
            cross[0:half-1, 0:half-1]                                   = imarray[0:half-1, 0:half-1]
            cross[0:half-1, half+1+gap:cross.shape[1]]                  = imarray[0:half-1, half+1:imarray.shape[1]]
            cross[half+1+gap:cross.shape[0], 0:half-1]                  = imarray[half+1:imarray.shape[0], 0:half-1]
            cross[half+1+gap:cross.shape[0], half+1+gap:cross.shape[1]] = imarray[half+1:imarray.shape[0], half+1:imarray.shape[1]]

#             cross = np.zeros((516, 516))
#             cross[0:255, 0:255] = imarray[0:255, 0:255]
#             cross[0:255, 261:516] = imarray[0:255, 257:512]
#             cross[261:516, 0:255] = imarray[257:512, 0:255]
#             cross[261:516, 261:516] = imarray[257:512, 257:512]
            valid_mask = (cross > 0)
            coords = np.array(np.nonzero(valid_mask)).T
            values = cross[valid_mask]
#             it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
            it = interpolate.NearestNDInterpolator(coords, values)
            filled = it(list(np.ndindex(cross.shape))).reshape(cross.shape)
        
            final.append(filled)
            print(fname)
      
        final = np.asarray(final) 
        return final      
    
def load_medipix3_files2(filepath='', filterstr='', flip = False, badpixelcorr = False, medianfilter = False, halvedata=False, fillgap=False):
    if os.path.exists(filepath):
        final = []
        cnt = 0
        for fname in tqdm(os.listdir(filepath)):
            filename, file_extension = os.path.splitext(fname)
            if len(filterstr) >0:
                if filename.find(filterstr)==-1:
                    continue
            if file_extension.lower() == '.tiff':
                if halvedata:
                    if cnt % 2 == 1:
                        cnt += 1
                        continue
                cnt += 1    
                im = Image.open(os.path.join(filepath, fname))
            else:
                continue
            imarray = np.array(im)
#             imarray[imarray < 0.07] = 0.0001
            
            if fillgap:
                gap = 2
                cross = np.zeros((imarray.shape[0]+gap, imarray.shape[1]+gap))
                half = np.int(imarray.shape[0]/2)
                cross[0:half, 0:half]                                   = imarray[0:half, 0:half]
                cross[0:half, half+gap:cross.shape[1]]                  = imarray[0:half, half:imarray.shape[1]]
                cross[half+gap:cross.shape[0], 0:half]                  = imarray[half:imarray.shape[0], 0:half]
                cross[half+gap:cross.shape[0], half+gap:cross.shape[1]] = imarray[half:imarray.shape[0], half:imarray.shape[1]]
                for j in range(0,cross.shape[0]):
                    value = cross[j,half-1]/2
                    cross[j, half-1:half+gap]=value
                    value = cross[j,half+gap]/2
                    cross[j,half-1+gap:half+1+gap]=value
                    value = cross[half-1,j]/2
                    cross[half-1:half+gap,j]=value
                    value = cross[half+gap,j]/2
                    cross[half-1+gap:half+1+gap,j]=value
                cross[256:258,256:258] = np.nan 
                imarray = cross

            if flip:
                imarray = np.flipud(imarray)
                imarray = np.fliplr(imarray)
            if badpixelcorr:
                imarray = imarray.astype(np.float64)
                imarray[imarray == np.inf] = np.nan
                imarray[imarray == -np.inf] = np.nan
                imarray[imarray == 0] = np.nan
                half = np.int(imarray.shape[0]/2)
                imarray[half-1] = np.nan
                imarray[half] = np.nan
                imarray[:,half-1] = np.nan
                imarray[:,half] = np.nan
                imarray = interpolate_replace_nans(imarray, kernel)#.astype(np.int16)
                

#                 valid_mask = (imarray > 0) & (imarray < 81900)
#                 coords = np.array(np.nonzero(valid_mask)).T
#                 values = imarray[valid_mask]
#                 it = interpolate.LinearNDInterpolator(coords, values, fill_value=1)
#     #             it = interpolate.NearestNDInterpolator(coords, values)
#                 imarray = it(list(np.ndindex(imarray.shape))).reshape(imarray.shape)

            if medianfilter:
#                 imarray = gaussian_filter(imarray, 7)
                imarray = median_filter(imarray, 7)
            
            final.append(imarray)
            
            
#             print(fname)
      
        final = np.asarray(final) 
        return final    
