"""
Modules designed to run slurm queues and enable differential evolution fitting

Copy-pasted by Camille Bishop, 18th Nov, 2022

"""


# imports
import shutil; import os
import MolecularGlass_Nov as MG
import numpy as np
from simple_slurm import Slurm



def reset_dir(target_dir):
    """
    A function to make a new temp directory to run CyRSoXS in
    """
    try:
        shutil.rmtree(target_dir)
    except OSError:
        pass
    os.mkdir(target_dir)
    
    
    

def runslurm_cb(running_dir, identifier, script_name, mdl_filename, result_filename, res=2048):
    """
    A function to get all of the files in the right place, run CyRSoXS through the slurm queue, then clean everything up.
    
    Parameters:
    
    running_dir: The parent directory everything runs in (e.g. Nov2022running; must have leading and end /)
    
    identifier: the identifier for the individual morphology being run. Combined with the running directory, this is a temp folder.
    
    script_name: .sh filename to make a script necessary to run CyRSoXS
    
    mdl_filename: .h5 filename of the morphology to run on
    
    result_filename: .nc file for xarray analysis
    
    res: resolution of image; in my case, it's always been 2048
    
    """
    
    
    
    os.chdir(f'{running_dir}{identifier}/')

    # writes an importing_python.py script with the directory names hardcoded in
    with open('importing_python.py','w') as f: 

        f.write(f'import sys\n')
        f.write(f'sys.path.insert(1, \047{running_dir}\047)\n')        
        f.write(f'from dean_reduce06 import \052\n')
        f.write('\n')        
        f.write(f'lfoo, lbar = cyrsoxs_datacubes(\047{running_dir}{identifier}/\047, 1.95, {res}, {res})\n')
        f.write('\n')
        f.write(f'print(lbar)\n')
        f.write(f'lbar.to_netcdf(\047{running_dir}{identifier}/{result_filename}\047)\n')

# ^^^^ Modify the above to write some sort of result file I'll want; I don't yet know what this will entail. Looks like dean used his reduce_06 code, then used "cyrsoxs_datacubes" to process the way he wanted
    
    #set up slurm job
    slurm = Slurm(
        gres='gpu:1',
        job_name= 'CamilleBishop',
        output=f'cyrsoxs.{Slurm.JOB_ARRAY_ID}.out',
        error = f'cyrsoxs.{Slurm.JOB_ARRAY_ID}.out',
        nodes = 1,
        ntasks = 1,
        ntasks_per_node = 1,
        export='ALL',
        gres_flags = 'enforce-binding',
        partition = 'gpu' # new that I just added at Dean's suggestion
#        gres_flags = 'disable-binding'
    )
    # Write a shell script for this job. Make sure that the 
    # script won't get overwritten before the job is run
    with open(script_name,'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')        
        f.write('source ~/.typyEnv CLEAN\n')
        f.write('\n')        
        f.write(f'typyEnv --add cyrsoxs/1.1.4.0\n')
        f.write('\n')
        # this next line should be modified to be the
        # CyRSoXS command (CyRSoXS_N2 input.h5)
        f.write(f'CyRSoXS {running_dir}/{identifier}/{mdl_filename}\n') 
        f.write(f'python {running_dir}/{identifier}/importing_python.py\n')
        
    # submit to queue!
    submitted = None
    while submitted is None:
        try:
            submitted = slurm.sbatch(f'bash {script_name}')
        except:
            pass

        
# string1 = 'therefore'
# string2 = 'it is'
# print(string1 + string2)
# print(f'{string1}_{string2}')        
        
      
def identifier_constructor(Tsub,param_tuple):
    """
    A function to create an identifier with all necessary info
    
    Parameters:
    
    Tsub: substrate temp; integer
    
    param_tuple: of form, e.g., (f'{hTPDnm}nm', f'{hDOnm}nm',...)
    
    """
    # no; needs a tuple of param strings
    # param_tuple = (parmstring1, parmstring2, etc.)
    num_params = np.size(param_tuple)
    i = 0
    identifier = f'{Tsub}K_'
    while i < num_params:
        if i == num_params - 1:
            identifier = identifier + param_tuple[i]
        else:
            identifier = identifier + param_tuple[i] + '_' # not using shortcut notation for clarity
        i += 1
        
    return identifier


## might eventually move this to another .py 
        
"""

Need to figure out how to make below work. It would be very ideal to have the sweep runner take in an analysis function of choice, rather than have it be set.

"""
    
    
    
# def run_a_sweep(running_directory,AFM_info,purity,sigma_blur_nm,noisefloor,outputPNGfolder,outputTXTfolder,Tsub,hTPD_nm,
#                   hDO_nm,STPD,SDO, analysis_func, theta_dist=True,minmax_theta=(60.,90.),theta_deg=90.,Eshift=1.,edge_on=True):
    
#     """
#     Function that will let you generate a morphology based on input parameters, then run it through CyRSoXS and an analysis function.
    
#     Non- self-explanatory parameters below.
    
#     Params:
    
#     analysis_func: A function of choice that will be used to 
    
#     """
    
    
#     AFM_file = AFM_info[0]; Tsub = AFM_info[1]; h_film = AFM_info[2]; hminus = AFM_info[3]; hplus = AFM_info[4];
#     block_size = AFM_info[5]; filter_size = AFM_info[6]; offset = AFM_info[7]; voxel_size = AFM_info[8]; res = AFM_info[9]

#     morph = MG.euc_morphology_20221109(AFM_info,purity,sigma_blur_nm,hTPD_nm,hDO_nm, STPD, SDO,add_OOP=True,theta_dist_shell=theta_dist, theta_dist_shell_minmax=minmax_theta,
#                                       default_theta_shell_deg=theta_dig,edge_on=edge_on)
    
#     #identifier = f'{Tsub}K_{round(hTPD_nm,2)}nmTPD_{round(hDO_nm,2)}nmDO_{round(purity,2)}%pure_{round(sigma_blur_nm,2)}nmBlur{theta_deg}theta'
    
#     identifier = f'{Tsub}K_{round(hTPD_nm,2)}nmTPD_{round(hDO_nm,2)}nmDO_{STPD}STPD_{SDO}SDO_{minmax_theta[0]}to{minmax_theta[1]}thetadist_face_on{face_on}'
    
#     ht.reset_dir(f'{running_directory}/{identifier}/')

#     EMG.save_hdf5(f'{running_directory}/{identifier}/{identifier}_glassy.h5', morph, 1.95, 3, mformat='ZYX')
        
#     os.symlink(f'{running_directory}/config.txt',f'{running_directory}/{identifier}/config.txt')
#     os.symlink(f'{running_directory}/Material1.txt',f'{running_directory}/{identifier}/Material1.txt')
#     os.symlink(f'{running_directory}/Material2.txt',f'{running_directory}/{identifier}/Material2.txt')
#     os.symlink(f'{running_directory}/Material3.txt',f'{running_directory}/{identifier}/Material3.txt')
    
#     ht.runslurm_cb(f'{running_directory}/{identifier}/',running_directory,'cbscript.sh',f'{identifier}_glassy.h5','temp.nc',2048)

#     mse = calcplot_Ivsq_Avsq_AvsE(NCfile=f'{running_directory}/{identifier}/temp.nc',energiesAvsq=[284.5,284.75,300.],energiesIvsq=[270.,284.5],Tsub=Tsub,identifier=identifier,h5_toplot=f'{running_directory}/{identifier}/{identifier}_glassy.h5',
#                                   noisefloor=0.,outputPNGfolder=outputPNGfolder,outputTXTfolder=outputTXTfolder,Eshift=Eshift)
        
#     try:
#         shutil.rmtree(f'{running_directory}/{identifier}')
#     except OSError:
#         pass
#     # deleted the directory
#     return mse