# Python script to allow for running of MGs_Euler-ZYX in cleaner format

import h5py
from matplotlib import gridspec, rc, colorbar; from matplotlib.pyplot import imshow, subplots
from skimage import io; from skimage.util import invert; from skimage.morphology import binary_dilation, binary_erosion, disk, diameter_closing, closing, opening, square, disk, skeletonize, thin, medial_axis, dilation, remove_small_holes
from simple_slurm import Slurm
import shutil; from PIL import Image; from io import BytesIO; import itertools; import xarray as xr; from time import sleep; from scipy import optimize; import warnings; import sys
import multiprocessing; from numba import jit; from matplotlib.ticker import FormatStrFormatter; import glob; from noise import snoise2
import matplotlib.cm as cm; from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import math; from copy import deepcopy; from matplotlib import colors as mcolors; import colorsys

import numpy as np; import matplotlib; import matplotlib.pyplot as plt; import datetime; from skimage import io; from skimage.filters import threshold_otsu, threshold_local, gaussian, try_all_threshold
from skimage.color import rgb2gray; from skimage.morphology import binary_dilation, disk, white_tophat, black_tophat; from skimage.feature import blob_dog, blob_log, blob_doh
import os
from matplotlib.patches import Rectangle

from skimage.morphology import disk, square; from skimage.filters import threshold_otsu, rank; from skimage.util import img_as_ubyte



# makes a binary image filter to assign Material 1 and Material 2 to morphologies
def process_image(AFM_data,block_size=91,filter_size=7, offset=0.007):
    image=AFM_data
    dim = image.shape[0]
    a_local_thresh = threshold_local(image,block_size=block_size, offset=offset)
    a_image_blur = gaussian(image, filter_size, output=None, mode='nearest')
    a  = a_image_blur < (a_local_thresh - offset)
    b = np.invert(a)
    return a,b #This spits out a, the morphology of the matrix, and b, the morphology of the holes. [BE CAREFUL WITH THIS; I think it may switch based on parameters. Always CheckH5.]

def process_imagenpy(AFM_data,block_size=91,filter_size=7, offset=0.007):
    #moon = io.imread(AFM_filename); image = rgb2gray(moon)
    # moon = np.load(AFM_filename)
    gray_image = AFM_data
    dim = gray_image.shape[0]
    a_local_thresh = threshold_local(gray_image,block_size=block_size, offset=offset)
    a_image_blur = gaussian(gray_image, filter_size, output=None, mode='nearest')
    a  = a_image_blur < (a_local_thresh - offset)
    b = np.invert(a)
    return a,b #This spits out a, the morphology of the matrix, and b, the morphology of the holes. [BE CAREFUL WITH THIS; I think it may switch based on parameters. Always CheckH5.]



# generates a 3D ZYX set of volume fractions for a given AFM and image processing parameters; set vacuum to True or False depending on including surface roughness   
def make_3D_MatVac_toggleZYX(AFM_data, pxx_nm,pxy_nm, h_film, hmin, hmax, block_size=81,filter_size=7,offset=0.003,vacuum=True): # add switch to turn off surface roughness
    # voxel_size = pxx_nm
    # moon = io.imread(AFM_filename); 
    gray_image = AFM_data
    h_difference = gray_image*np.abs(hmax-hmin)
    softmatter_3D = np.zeros([int(np.rint(h_film/voxel_size)),np.shape(gray_image)[1],np.shape(gray_image)[0]],dtype=bool) # z, y, x
    min_thick_nm = h_film - (hmax-hmin) # film thickness at valleys in nm
    for i in range(0,np.shape(softmatter_3D)[2]): # x
        for j in range(0,np.shape(softmatter_3D)[1]): # y
            voxel_num = int(np.rint((min_thick_nm + h_difference[i,j])/voxel_size))
            # now run through this number of voxels on this x,y coordinate, and fill in ones for this many voxels
            for k in range(0,voxel_num):
                softmatter_3D[k,j,i] = True
    vac_boolMorph = np.copy(softmatter_3D); vac_boolMorph = np.invert(vac_boolMorph)
    int_VacMorph = vac_boolMorph*1
    a,b = process_image(AFM_filename,block_size,filter_size,offset)
    int_AMorph = np.copy(softmatter_3D) # boolean
    int_BMorph = np.copy(softmatter_3D)
    for i in range(0,np.shape(vac_boolMorph)[0]):
        int_AMorph[i,:,:] = int_AMorph[i,:,:]*a
        int_BMorph[i,:,:] = int_BMorph[i,:,:]*b
    if vacuum == False:
        # want to chop off all of the layers above the minimum thickness
        valley_ht_vox = int(np.rint((min_thick_nm/voxel_size)))
        int_AMorph = int_AMorph[0:valley_ht_vox,:,:]
        int_BMorph = int_BMorph[0:valley_ht_vox,:,:]
        int_VacMorph = int_AMorph*0
    return int_VacMorph, int_AMorph, int_BMorph

def make_3D_MatVac_toggleZYXnpy(AFM_filename, pxx_nm,pxy_nm, h_film, hmin, hmax, block_size=81,filter_size=7,offset=0.003,vacuum=True): # add switch to turn off surface roughness
    voxel_size = pxx_nm
#     moon = io.imread(AFM_filename);gray_image = rgb2gray(moon)
    moon = np.load(AFM_filename)
    gray_image = rgb2gray(moon)
    h_difference = gray_image*np.abs(hmax-hmin)
    softmatter_3D = np.zeros([int(np.rint(h_film/voxel_size)),np.shape(gray_image)[1],np.shape(gray_image)[0]],dtype=bool) # z, y, x
    min_thick_nm = h_film - (hmax-hmin) # film thickness at valleys in nm
    for i in range(0,np.shape(softmatter_3D)[2]): # x
        for j in range(0,np.shape(softmatter_3D)[1]): # y
            voxel_num = int(np.rint((min_thick_nm + h_difference[i,j])/voxel_size))
            # now run through this number of voxels on this x,y coordinate, and fill in ones for this many voxels
            for k in range(0,voxel_num):
                softmatter_3D[k,j,i] = True
    vac_boolMorph = np.copy(softmatter_3D); vac_boolMorph = np.invert(vac_boolMorph)
    int_VacMorph = vac_boolMorph*1
    a,b = process_imagenpy(AFM_filename,block_size,filter_size,offset)
    int_AMorph = np.copy(softmatter_3D) # boolean
    int_BMorph = np.copy(softmatter_3D)
    for i in range(0,np.shape(vac_boolMorph)[0]):
        int_AMorph[i,:,:] = int_AMorph[i,:,:]*a
        int_BMorph[i,:,:] = int_BMorph[i,:,:]*b
    if vacuum == False:
        # want to chop off all of the layers above the minimum thickness
        valley_ht_vox = int(np.rint((min_thick_nm/voxel_size)))
        int_AMorph = int_AMorph[0:valley_ht_vox,:,:]
        int_BMorph = int_BMorph[0:valley_ht_vox,:,:]
        int_VacMorph = int_AMorph*0
    return int_VacMorph, int_AMorph, int_BMorph

# generates a morphology ready to write to an hdf5 that is unaligned; in practice, will rarely use this moving forward
def unaligned_Morphology(AFM_filename, pix_nm, h_film, hmin, hmax, block_size=81, filter_size=7, offset=0.003, vacuum=True):
    Vfrac_vac, Vfrac_A, Vfrac_B = make_3D_MatVac_toggleZYX(AFM_filename, pix_nm,pix_nm, h_film, hmin, hmax, block_size=81,filter_size=7,offset=0.003,vacuum=True)
    A_S = np.zeros_like(Vfrac_A); B_S = np.zeros_like(Vfrac_B); vac_S = np.zeros_like(Vfrac_vac)
    # Euler angles
    A_theta = np.zeros_like(Vfrac_A); B_theta = np.zeros_like(Vfrac_B); vac_theta = np.zeros_like(Vfrac_vac)
    A_psi = np.zeros_like(Vfrac_A); B_psi = np.zeros_like(Vfrac_B); vac_psi = np.zeros_like(Vfrac_vac)

    Euler_morph = [Vfrac_A, A_S, A_theta, A_psi,Vfrac_B, B_S, B_theta, B_psi, Vfrac_vac, vac_S, vac_theta, vac_psi] # has 12 components for a 3-material system; even without vacuum, since that will just be 0s
    return Euler_morph


# outputs the local arrays that compose morphology, so that they can be plotted
def unaligned_Morphology_Morph3D(AFM_filename, pix_nm, h_film, hmin, hmax, block_size=81, filter_size=7, offset=0.003, vacuum=True,invert=False):
    Vfrac_vac, Vfrac_A, Vfrac_B = make_3D_MatVac_toggleZYX(AFM_filename, pix_nm,pix_nm, h_film, hmin, hmax, block_size=81,filter_size=7,offset=0.003,vacuum=True)
    A_S = np.zeros_like(Vfrac_A); B_S = np.zeros_like(Vfrac_B); vac_S = np.zeros_like(Vfrac_vac)
    # Euler angles
    A_theta = np.zeros_like(Vfrac_A); B_theta = np.zeros_like(Vfrac_B); vac_theta = np.zeros_like(Vfrac_vac)
    A_psi = np.zeros_like(Vfrac_A); B_psi = np.zeros_like(Vfrac_B); vac_psi = np.zeros_like(Vfrac_vac)
    if invert == False: # results in Material A in the holes
        Euler_morph = [Vfrac_A, A_S, A_theta, A_psi,Vfrac_B, B_S, B_theta, B_psi, Vfrac_vac, vac_S, vac_theta, vac_psi] # has 12 components for a 3-material system; even without vacuum, since that will just be 0s
    if invert == True: #results in Material B in the holes
        Vfrac_Ainv = np.copy(Vfrac_B); A_Sinv = np.copy(B_S); A_thetainv = np.copy(B_thetainv); A_psiinv = np.copy(B_psiinv)
        Vfrac_Binv = np.copy(Vfrac_A); B_Sinv = np.copy(A_S); B_thetainv = np.copy(A_thetainv); B_psiinv = np.copy(A_psiinv)
        Euler_morph = [Vfrac_Ainv, A_Sinv, A_thetainv, A_psiinv,
                       Vfrac_Binv, B_Sinv, B_thetainv, B_psiinv,
                       Vfrac_vac, vac_S, vac_theta, vac_psi]
        Vfrac_A = Vfrac_Ainv; Vfrac_B = Vfrac_Binv
    return Euler_morph, Vfrac_vac, Vfrac_A, Vfrac_B
    


# plots a morphology generated by "make_3D_MatVac_toggleZYX" in 3D in the notebook; note that this takes a while, so choose a small slice (built-in sort of safeguard)
def plot_morphcomponentZYX(Morph3D, x_min_vox, x_max_vox, y_min_vox, y_max_vox, z_min_vox, z_max_vox, I_understand=False): # includes a parameter so you get warned that this will take a long time
    if I_understand==False:
        print('Please agree to understanding that this function may take very long to execute if a large slice is taken, by modifying last parameter to True')
    else:
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(Morph3D[z_min_vox:z_max_vox,y_min_vox:y_max_vox,x_min_vox:x_max_vox])
        plt.show()

        
        
# use to calculate the volume fractions of the two components based upon the image processing parameters; haven't used since Boise
def find_vol_fracts(AFM_filename, pxx_nm,pxy_nm, h_film, AFM_hmin, AFM_hmax, block_size=91,filter_size=7,offset=0.007,TPDholebool=True): # need an input image of greyscale, and the thresholded image - want to make sure the threshold parameters are the same as that used to make the h5
    moon = io.imread(AFM_filename); 
    gray_image = rgb2gray(moon)
    h_difference = gray_image*np.abs(hmax-hmin)
    V_pix = pxx_nm*pxy_nm*(film_thick - np.abs(hmax-hmin)/2 + h_difference)
    holes_bool, matrix_bool = process_image(filename,block_size, filter_size, offset) # returns two boolean arrays; Trues for holes in the first, Trues for matrix in the 2nd
    V_mat_holes = 0.; V_mat_matrix = 0.;
    for i in range(0,np.shape(holes_bool)[0]):
        for j in range(0,np.shape(holes_bool)[1]):
            if holes_bool[i,j] == True:
                V_mat_holes = V_mat_holes + V_pix[i,j]
            elif holes_bool[i,j] == False:
                V_mat_matrix = V_mat_matrix + V_pix[i,j]
    return V_mat_holes, V_mat_matrix


# generates a 3D ZYX set of volume fractions for a given AFM and image processing parameters; generally, use the similar "toggle" function with vacuum set to  True        
def make_3D_MatVacZYX(AFM_filename, pxx_nm,pxy_nm, h_film, hmin, hmax, block_size=81,filter_size=7,offset=0.003):
    voxel_size = pxx_nm
    moon = io.imread(AFM_filename); gray_image = rgb2gray(moon)
    h_difference = gray_image*np.abs(hmax-hmin)
    softmatter_3D = np.zeros([int(np.rint(h_film/voxel_size)),np.shape(gray_image)[1],np.shape(gray_image)[0]],dtype=bool) # z, y, x
    min_thick_nm = h_film - (hmax-hmin) # film thickness at valleys in nm
    for i in range(0,np.shape(softmatter_3D)[2]): # x
        for j in range(0,np.shape(softmatter_3D)[1]): # y
            voxel_num = int(np.rint((min_thick_nm + h_difference[i,j])/voxel_size))
            # now run through this number of voxels on this x,y coordinate, and fill in ones for this many voxels
            for k in range(0,voxel_num):
                softmatter_3D[k,j,i] = True
    vac_boolMorph = np.copy(softmatter_3D); vac_boolMorph = np.invert(vac_boolMorph)
    int_VacMorph = vac_boolMorph*1
    a,b = process_image(AFM_filename,block_size,filter_size,offset)
    int_AMorph = np.copy(softmatter_3D) # boolean
    int_BMorph = np.copy(softmatter_3D)
    for i in range(0,np.shape(vac_boolMorph)[0]):
        int_AMorph[i,:,:] = int_AMorph[i,:,:]*a
        int_BMorph[i,:,:] = int_BMorph[i,:,:]*b
    int_AMorph = int_AMorph*1; int_BMorph = int_BMorph*1
    return int_VacMorph, int_AMorph, int_BMorph

# should really generalize this to materials A & B
def save_hdf5(filename, morph, voxel_size, num_mat, mformat='ZYX'): # morph needs to hold 9 3x3 matrices; need a line in the function to specify the components
    # morph now has 4xnum_mat components
    # where do I specify XYZ?
    A_vfrac = morph[0]; A_S = morph[1]; A_theta = morph[2]; A_psi = morph[3]
    B_vfrac = morph[4]; B_S = morph[5]; B_theta = morph[6]; B_psi = morph[7]
    vac_vfrac = morph[8]; vac_S = morph[9]; vac_theta = morph[10]; vac_psi = morph[11]
    
    with h5py.File(f'{filename}', "w") as f:
        f.create_dataset('igor_parameters/igorefield', data="0,1")
        f.create_dataset('igor_parameters/igormaterialnum', data=3)
        f.create_dataset('igor_parameters/igornum', data='')
        f.create_dataset('igor_parameters/igorthickness', data='')
        f.create_dataset('igor_parameters/igorvoxelsize', data='')

        f.create_dataset('Morphology_Parameters/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")) # datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/film_normal', data=[1,0,0])
        f.create_dataset('Morphology_Parameters/morphology_creator', data='')
        f.create_dataset('Morphology_Parameters/name', data='')
        f.create_dataset('Morphology_Parameters/version', data='')#data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        f.create_dataset('Morphology_Parameters/PhysSize', data=voxel_size)
        f.create_dataset('Morphology_Parameters/NumMaterial',data=num_mat)
        f.create_dataset('Morphology_Parameters/Parameters',data='')
        f.create_dataset('Morphology_Parameters/Parameter_values',data='')

        mat1vf = f.create_dataset('Euler_Angles/Mat_1_Vfrac', data=A_vfrac.astype(np.float64))
        mat1s = f.create_dataset('Euler_Angles/Mat_1_S', data=A_S.astype(np.float64))
        mat1theta = f.create_dataset('Euler_Angles/Mat_1_Theta', data=A_theta.astype(np.float64))
        mat1psi = f.create_dataset('Euler_Angles/Mat_1_Psi', data=A_psi.astype(np.float64))
        mat2vf = f.create_dataset('Euler_Angles/Mat_2_Vfrac', data=B_vfrac.astype(np.float64))
        mat2s = f.create_dataset('Euler_Angles/Mat_2_S', data=B_S.astype(np.float64))
        mat2theta = f.create_dataset('Euler_Angles/Mat_2_Theta', data=B_theta.astype(np.float64))
        mat2psi = f.create_dataset('Euler_Angles/Mat_2_Psi', data=B_psi.astype(np.float64))  
        mat3vf = f.create_dataset('Euler_Angles/Mat_3_Vfrac', data=vac_vfrac.astype(np.float64))
        mat3s = f.create_dataset('Euler_Angles/Mat_3_S', data=np.zeros_like(vac_vfrac).astype(np.float64))
        mat3theta = f.create_dataset('Euler_Angles/Mat_3_Theta', data=np.zeros_like(vac_vfrac).astype(np.float64))
        mat3psi = f.create_dataset('Euler_Angles/Mat_3_Psi', data=np.zeros_like(vac_vfrac).astype(np.float64)) # will always be zeros for vacuum, so why not
        for i in range(1,4):
            for j in range(0,3):
                locals()[f"mat{i}vf"].dims[j].label = mformat[j]
                locals()[f"mat{i}s"].dims[j].label = mformat[j]
                locals()[f"mat{i}theta"].dims[j].label = mformat[j]
                locals()[f"mat{i}psi"].dims[j].label = mformat[j]
        f.close()