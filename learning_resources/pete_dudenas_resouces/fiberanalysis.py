import numpy as np
import matplotlib.pyplot as plt
# import pySPM

from skimage.morphology import skeletonize,binary_erosion, binary_dilation, binary_opening, binary_closing,square, thin, remove_small_objects, medial_axis
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.feature import canny

from scipy.signal import convolve

def bigshow(array,dpi=140,colorbar=False,**kwargs):
    plt.figure(dpi=dpi)
    plt.imshow(array,**kwargs)
    if colorbar:
        plt.colorbar()

def readcorrect(file,channel='Phase'):
    ''' Reads phase data from Bruker AFM scan file
    and aligns rows by subtracting median from each row
    '''
    scan = pySPM.Bruker(file)
    phase = np.array(scan.get_channel(channel).pixels)
    for i,row in enumerate(phase):
        phase[i] -= np.median(row)
    return phase

def fiber_sort(xycoords,startxy):
    '''sorts list from one end to the other.
    used in conjunction with label to sort
    fiber from one end to the other'''
    tobesorted = xycoords.copy()
#     xysorted = np.zeros_like(xycoords)
    sorted_coords = []
    reference_point = startxy

    while len(tobesorted) > 0:
        dist = distance(tobesorted,reference_point)
        next_shortest = np.argmin(dist)
        sorted_coords.append(tobesorted[next_shortest])
        reference_point = tobesorted[next_shortest]
        tobesorted = np.delete(tobesorted,next_shortest,axis=0)
    return np.array(sorted_coords)

def smooth_fiber_core(xlist, ylist, avg_width,repeat=0):
    ''' Smooths out grow_fiber_core results using a
    moving window average of width avg_width.
    Used for axial director calculation '''
    if avg_width > len(xlist)/2:
        avg_width = int(len(xlist)/2)
    xold = xlist.copy()
    yold = ylist.copy()
    for i in range(repeat+1):
        xsmooth = []
        ysmooth = []

        for j in np.arange(0,avg_width):
            xsmooth.append(np.mean(xold[0:(2*j+1)]))
            ysmooth.append(np.mean(yold[0:(2*j+1)]))
        for j in np.arange(avg_width,len(xold)-avg_width):
            min_idx = j-avg_width
            max_idx = j+avg_width+1
            xsmooth.append(np.mean(xold[min_idx:max_idx]))
            ysmooth.append(np.mean(yold[min_idx:max_idx]))
        for j in np.arange(len(xold)-avg_width,len(xold)):
            xsmooth.append(np.mean(xold[j:]))
            ysmooth.append(np.mean(yold[j:]))

        xold = xsmooth.copy()
        yold = ysmooth.copy()

    return xsmooth, ysmooth

def distance(array, position):
    return np.sqrt(np.sum((array-position)**2,axis=1))


def remove_branches(skeleton_array,square_size=3):
    ''' Removes branch points of skeletonized arrays
    so that fiber_sort works correctly
    '''
    struct_elem = square(square_size)
    convolved_array = convolve(skeleton_array,struct_elem,mode='same')
    branch_points = convolved_array>3
    skeleton_array[branch_points] = 0
    isolated_points = convolved_array <= 1
    skeleton_array[isolated_points] = 0
    return skeleton_array


def calc_fibertheta(labeled_array,avg_width=2,repeat=3):
    '''calculates the axial director at each point along fibers
    in a labeled image. Automatically smooths fiber coordinates'''
    theta_out = []
    for region in regionprops(labeled_array):
        if len(region.coords) > 6:
            xy_coords = np.array(sorted(region.coords,key=lambda y: y[0]))

            # find farthest point from center of fiber - assume this is one end
            r = np.sqrt(np.sum((region.coords-region.centroid)**2,axis=1))
            far = np.argmax(r)
            coords = fiber_sort(xy_coords,region.coords[far])

            # smooth out fibers
            xsmooth,ysmooth = smooth_fiber_core(coords[:,1].astype(float),coords[:,0].astype(float),avg_width,repeat)
            theta = np.arctan(np.diff(ysmooth)/np.diff(xsmooth))
            for val in theta:
                theta_out.append(val)
    return theta_out


def create_smooththeta(labeled_array,avg_width=2,repeat=3):
    ''' takes a labeled array and returns a numpy array
    with the smoothed fiber axial director assigned for each pixel.
    Pixels with no fiber are assigned to np.nan'''
    smooth_img = np.zeros(labeled_array.shape)
    smooth_img[:] = np.nan
    for region in regionprops(labeled_array):
        if len(region.coords) > 3:
            xy_coords = np.array(sorted(region.coords,key=lambda y: y[0]))

            # find farthest point from center of fiber - assume this is one end
            r = np.sqrt(np.sum((region.coords-region.centroid)**2,axis=1))
            far = np.argmax(r)
            coords = fiber_sort(xy_coords,region.coords[far])

            # smooth out fibers
            xsmooth,ysmooth = smooth_fiber_core(coords[:,1].astype(float),coords[:,0].astype(float),avg_width,repeat)
#             print(ysmooth)
            theta = np.arctan(np.diff(ysmooth)/np.diff(xsmooth))
            theta_out = np.zeros(len(xsmooth))
            theta_out[0] = theta[0]
            theta_out[1:] = theta[:]
#             print(theta_out)
            for i, val in enumerate(theta_out):
#                 print(val)
                smooth_img[int(ysmooth[i]),int(xsmooth[i])] = val
    return smooth_img
