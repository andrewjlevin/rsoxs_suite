""" Molecular Glass Morphology Generation
This is code updated in November 2022 to include out-of-plane orientation, with
full distributions

Author: Camille Bishop

Created: 18th November, 2022"""

# imports
# double check which of these are actually needed
import sys

import numpy as np; from skimage import io; from skimage.filters import threshold_local, gaussian
from skimage.color import rgb2gray; from skimage.feature import blob_dog, blob_log, blob_doh
import os

import random; np.random.seed(10)
# put a seed in

import skimage
import datetime
from scipy.ndimage.morphology import distance_transform_edt

import shutil; from PIL import Image; from io import BytesIO; import itertools; import xarray as xr; from time import sleep; from scipy import optimize; import warnings; import sys
import multiprocessing; from numba import jit; from matplotlib.ticker import FormatStrFormatter; import glob; from noise import snoise2
import math; from copy import deepcopy; import colorsys
import imageio
import h5py
import warnings

# modules
def euc_morphology_20221109(AFM_info, purity, sigma_blur_nm, hTPD_nm, hDO_nm, S_TPD, S_DO,add_OOP=True,theta_dist_shell=True,
                            theta_dist_shell_minmax=(60.,90.),default_theta_shell_deg=90.,edge_on=True):
    """
    The most recent morphology generation as of mid-November, which includes out-of-plane orientation drawn from GIXD distributions.
    
    An important change: it's so clunky to put all the AFM parameters in a function, I am defining a new variable "AFM_information", which is a tuple that contains:
    
    AFM_info: (AFM_file, Tsub, h_film, hminus, hplus, block_size, filter_size, offset, voxel_size, res)
    
    purity: the parameter that defines how much mixing there is between domains. I could modify this to account for the equilibrium separation
    
    sigma_blur_nm: determines the blurring between domains. Need to be very careful where nm is used, and voxels - the gaussian filter needs it in voxels.
    
    hTPD_nm, hDO_nm: thickness of the oriented regions. Once again, most equations will need it in voxels. Make sure to not double count!
    
    S_TPD, S_DO: order parameters for the oriented regions
    
    add_OOP: should generally be set to True, sets out of plane orientation within TPD domains
    
    theta_dist_shell: determines whether to use a distribution of theta angles within the shells, rather than a single static angle.
    Right now, it is drawn randomly from a range; it should probably be changed to a normal distribution.
    
    theta_dist_shell_minmax: if a theta distribution is used, determines the range over which
    
    default_theta_shell_deg: if a distribution is not used, the angle theta is set to
    
    edge-on: whether the molecules are oriented edge-on to the interface. Should usually be set to true, but exists just in case you're doing hypothesis testing.
    
    """
    AFM_file = AFM_info[0]; Tsub = AFM_info[1]; h_film = AFM_info[2]; hminus = AFM_info[3]; hplus = AFM_info[4];
    block_size = AFM_info[5]; filter_size = AFM_info[6]; offset = AFM_info[7]; voxel_size = AFM_info[8]; res = AFM_info[9]
    
    isotropic_morph, hole_binary, matrix_binary = isotropic_morph_2D_purityblur(AFM_info,sigma_blur_nm,purity)
    
    hTPDvox = hTPD_nm / 1.95; hDOvox = hDO_nm / 1.95
    
    Vfrac_A = np.copy(isotropic_morph[0]); Snew_A = np.copy(isotropic_morph[1]); thetanew_A = np.copy(isotropic_morph[2]); psinew_A = np.copy(isotropic_morph[3])
    Vfrac_B = np.copy(isotropic_morph[4]); Snew_B = np.copy(isotropic_morph[5]); thetanew_B = np.copy(isotropic_morph[6]); psinew_B = np.copy(isotropic_morph[7])

    # the shell needs to be determined from a binary, and the isotropic morphology may have a blur to it with fractional numbers
    a_med = hole_binary; b_med = matrix_binary 
    dist_holes = scipy_edt(hole_binary)
    dist_mat = scipy_edt(matrix_binary)
    a_shell = (dist_holes < 0) * (dist_holes > -hTPDvox)
    b_shell = (dist_mat < 0) * (dist_mat > -hDOvox) # the regions which will be oriented


    a_mod = np.expand_dims(a_med,axis=0); b_mod = np.expand_dims(b_med,axis=0)
    a_shell_mod = np.expand_dims(a_shell, axis=0); b_shell_mod = np.expand_dims(b_shell, axis=0)
    a_gauss = gaussian(a_mod[0,:,:]*1.0,3,output=None,mode='nearest'); b_gauss = gaussian(b_mod[0,:,:]*1.0,3,output=None,mode='nearest')
    a_grad = np.gradient(a_gauss); b_grad = np.gradient(b_gauss)
    # choosing if there's a distribution of theta values in the shell (should be ~>60)
    if theta_dist_shell==True:
        shell_theta_dist = np.random.uniform(theta_dist_shell_minmax[0]*np.pi/180.,theta_dist_shell_minmax[1]*np.pi/180,np.shape(a_med))
        # may be better to choose a gaussian distribution here peaked at a certain value
        
    #Using literature GIXD to assign out of plane orientation to the bulk TPD
    if add_OOP==True:
        if Tsub == 280:
            distribution = np.loadtxt('/home/ceb10/ExperimentalData/TPDGIWAXS_Integrations_Ediger/260K_polyfit_probabilities.txt') # this wasn't working within the notebook, so may have to fine-tune
            bulk_theta_dist = random.choices(distribution[:,0],weights=distribution[:,1],k=2048**2)
            bulk_theta_dist= (np.pi/180)*np.reshape(bulk_theta_dist,(2048,2048))
            S_exp = 0.10
        elif Tsub == 310:
            distribution = np.loadtxt('/home/ceb10/ExperimentalData/TPDGIWAXS_Integrations_Ediger/315K_polyfit_probabilities.txt')
            bulk_theta_dist = random.choices(distribution[:,0],weights=distribution[:,1],k=2048**2)
            bulk_theta_dist = (np.pi/180)*np.reshape(bulk_theta_dist,(2048,2048))
            S_exp = 0.05
        elif Tsub == 325:
            add_OOP = False # there is no out of plane orientation in this case
        # if there is out-of-plane orientation, need to introduce randoms for the psis to avoid correlations
        bulk_psi = np.random.uniform(0.,2*np.pi,np.shape(a_med)) # would be interesting to check with neat TPD that there aren't orientational correlations....
    

    for i in range(0,np.shape(a_shell)[0]): # all x
        for j in range(0,np.shape(a_shell)[1]): # all y
            if a_shell[i,j] == True: # if something is part of the TPD shell, (DO37 or TPD)
                Snew_A[0,i,j] = S_TPD # assign it to the S that is part of the shell
                Snew_B[0,i,j] = S_DO # basically, any oriented DO37 should have the same orientation on both sides of the shell. Volume fraction will take care of it
                if theta_dist_shell==True:
                    thetanew_A[0,i,j]=shell_theta_dist[i,j] # taking care of the distribution of theta values in the shell ("random" is maybe not the best word, it's pulled from a distribution)
                    thetanew_B[0,i,j]=shell_theta_dist[i,j]
                else:
                    thetanew_A[0,i,j] = default_theta_shell_deg*np.pi/180. # We are just assuming that all of the non-shell molecules are oriented with pi systems completely parallel to substrate
                    thetanew_B[0,i,j] = default_theta_shell_deg*np.pi/180. # We are just assuming that all of the non-shell molecules are oriented with pi systems completely parallel to substrate
                if edge_on==False:
                    psinew_A[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) # keep the same radial orientation
                    psinew_B[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) 
                else:
                    psinew_A[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) + np.pi/2 # keep the same radial orientation
                    psinew_B[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) + np.pi/2# this is looking for DO blended into the TPD shell - It gets to have the same orientation as the DO37 shell, which I think is correct
            if b_shell[i,j] == True: # if something is part of the DO37 shell
                Snew_B[0,i,j] = S_DO # assign it to the S that is part of the shell
                Snew_A[0,i,j] = S_TPD
                if theta_dist_shell==True:
                    thetanew_B[0,i,j]=shell_theta_dist[i,j]
                    thetanew_A[0,i,j]=shell_theta_dist[i,j]
                else:
                    thetanew_A[0,i,j] = default_theta_shell_deg*np.pi/180. # We are just assuming that all of the non-shell molecules are oriented with pi systems completely parallel to substrate
                    thetanew_B[0,i,j] = default_theta_shell_deg*np.pi/180.
                if edge_on==False:
                    psinew_A[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) # keep the same radial orientation
                    psinew_B[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) 
                else:
                    psinew_A[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) + np.pi/2 # keep the same radial orientation
                    psinew_B[0,i,j] = np.arctan2(a_grad[0][i,j],a_grad[1][i,j]) + np.pi/2
            elif (a_med[i,j]==True) and (a_shell[i,j]==False): # if it's in the TPD composition, but is not included in either of the interfacial regions (maybe this is the key I was missing)
                if add_OOP==True: # now we need to assign it to a random psi and a theta drawn from the GIXD distribution
                    Snew_A[0,i,j] = S_exp
                    thetanew_A[0,i,j] = bulk_theta_dist[i,j]
                    psinew_A[0,i,j] = bulk_psi[i,j]
                
            else: # it's not in either of the shells or in the bulk TPD
                pass               
           
    return [Vfrac_A, Snew_A, thetanew_A, psinew_A, Vfrac_B, Snew_B, thetanew_B, psinew_B, isotropic_morph[8], isotropic_morph[9], isotropic_morph[10], isotropic_morph[11]]


def scipy_edt(b): # this takes a boolean array, I believe
    """
    Uses euclidean distance, rather than old dilate/erode, to calculate distances from interfaces. Updated by team ~ Sep 2022
    
    Parameter b is a boolean array for a binarized morphology.
    """
    d_x = b.shape[0]
    d_y = b.shape[1] # x and y dimensions of the array
    dist = np.zeros([d_x,d_y]).astype(float) # setting up 0s to hold the distances
#    dist = 0.5-distance_transform_edt(b)
    dist[np.where(b==True)] = 0.5-distance_transform_edt(b)[np.where(b==True)]
    dist[np.where(b==False)] = distance_transform_edt(b==False)[np.where(b==False)] -0.5

    return dist


# Generate an isotropic morphology with a blur and specified purity (purity parameter may need to be adjusted)
def isotropic_morph_2D_purityblur(AFM_info, sigma_nm=0., purity=100.):
    '''
    Calculates a morphology without interfacial or out-of-plane orientation.
    
    Parameters:
    
    AFM_info: tuple consisting of (AFM_file, Tsub, h_film, hminus, hplus, block_size, filter_size, offset, voxel_size, res)
    
    sigma_nm: takes a sigma blur in nm
    
    purity: self-explanatory
    
    Returns: morph_to_save, hole_binary, matrix_binary
    
    
    '''
    AFM_file = AFM_info[0]; Tsub = AFM_info[1]; h_film = AFM_info[2]; hminus = AFM_info[3]; hplus = AFM_info[4];
    block_size = AFM_info[5]; filter_size = AFM_info[6]; offset = AFM_info[7]; voxel_size = AFM_info[8]; res = AFM_info[9]
    
    holes,matrix,gray_image = process_image(AFM_file,block_size,filter_size,offset) # made filters for what is material a and what is mat b
    
    sigma_vox = sigma_nm / 1.95
    # blur holes first
    holes_blur = skimage.filters.gaussian(holes*1., sigma=(sigma_vox, sigma_vox), truncate=3., multichannel=False) # turns a boolean into a np, by the way
    matrix_blur = np.subtract(np.ones(np.shape(holes_blur),dtype=float),holes_blur)

    TPD_hole_pct = (50. + 0.5*purity)/100.; TPD_matrix_pct = (50 - 0.5*purity)/100. # turns the purity parameter into percents in hole, matrix
    DO_hole_pct = (50 - 0.5*purity)/100.; DO_matrix_pct = (50 + 0.5*purity)/100.
    
    # remember I may need to re-think purity parameter; mass balance with heights as well as the 2D surface area
    holes_TPD = holes_blur*TPD_hole_pct; holes_DO = holes_blur*DO_hole_pct
    matrix_TPD = matrix_blur*TPD_matrix_pct; matrix_DO = matrix_blur*DO_matrix_pct

    h_max = h_film+np.abs(hplus); h_min = h_film-np.abs(hminus) # accounting for vacuum fractions
    heights = (gray_image*(np.abs(hplus)+np.abs(hminus))+(h_min))       
    Vfrac_V = (h_max - heights)/h_max #volume fraction of each voxel that is vacuum
    Vfrac_M = 1. - Vfrac_V # this will turn into numbers like 0.9 - so that is the pctg of material in it. Can use that as a scale factor for the 

    all_TPD = np.add(holes_TPD,matrix_TPD); all_DO = np.add(holes_DO,matrix_DO)

    TPD_adjusted = np.multiply(Vfrac_M,all_TPD); DO_adjusted = np.multiply(Vfrac_M,all_DO)

    Vfrac_TPD = np.expand_dims(TPD_adjusted,axis=0); Vfrac_DO = np.expand_dims(DO_adjusted, axis=0) # expanding dims for compatibility with CyRSOXS (2048x2048 to 1x2048x2048)
    Vfrac_V = np.expand_dims(Vfrac_V,axis=0)

    zeros_mat = np.zeros(np.shape(Vfrac_V),dtype=float)
    
    hole_binary = holes; matrix_binary = matrix # these outputs are necessary for adding orientation later; however, this way, we only have to generate the isotropic morphology once

    morph_to_save = [Vfrac_TPD, zeros_mat, zeros_mat, zeros_mat, Vfrac_DO, zeros_mat, zeros_mat, zeros_mat, Vfrac_V, zeros_mat, zeros_mat, zeros_mat]
    return morph_to_save, hole_binary, matrix_binary

def process_image(AFM_filename,block_size=91,filter_size=7, offset=0.007):
    moon = io.imread(AFM_filename); image = rgb2gray(moon)
    dim = image.shape[0]
    a_local_thresh = threshold_local(image,block_size=block_size, offset=offset)
    a_image_blur = gaussian(image, filter_size, output=None, mode='nearest')
    a  = a_image_blur < (a_local_thresh - offset)
    b = np.invert(a)
    gray_image = image
    return a,b,gray_image #This spits out a, the morphology of the matrix, and b, the morphology of the holes. [BE CAREFUL WITH THIS; I think it may switch based on parameters. Always CheckH5.]

def save_hdf5(filename, morph, voxel_size, num_mat, mformat='ZYX'): # morph needs to hold 9 3x3 matrices; need a line in the function to specify the components
    '''
    Saves a morphology to H5; taken from EulerMGs.py so that I no longer need to use that .py file
    '''
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