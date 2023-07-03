"""
Generate Cahn-Hilliard eqn. based morphologies using the fipy package:

Output: folder for each generated morphology named based on chosen parameters:
            hdf5 morphology (compatible with cyrsoxs)
            material#.txt files for each material (compatible with cyrsoxs)
            config.txt file (compatible with cyrsoxs)
            raw numpy arr data (.txt or .npy for 3D)
            png images of the generated morphology
            checkh5 pdf image of generated morphology (3D compat.?)
"""

### Imports:
import pathlib
from fipy import CellVariable, Grid3D, Viewer, GaussianNoiseVariable, Grid2D, TransientTerm, DiffusionTerm, DefaultSolver
from fipy.tools import numerix
import numpy as np
import matplotlib.pyplot as plt
import time

### Define paths & other variables:
basePath = pathlib.Path('/pl/active/Toney-group/anle1278/rsoxs_suite')
savePath = basePath.joinpath('imgs_analysis/sim_runs')
cmap = plt.cm.YlGnBu_r.copy()  # Set a sequential colormap


def gen_2d(nxy=200, dxy=1.5, mean=0.5, D=1., a=1., eps=1., steps=500, savePath=savePath, counter=1):
    """
    Generate a 2D BHJ morphology based on the Cahn-Hilliard Eqn. using a generic free energy term that penalizes 
    values of phi between 0 and 1. 

    Inputs: nxy: number of pixels 
            dxy: size of each pixel
            mean: starting average value for the generated 2D mesh
            D, a, eps: terms for the Cahn-Hilliard Eqn. (Diffusion Rate, free energy term, deg. of mixing term)
            steps: number of time steps to simulate
            savePath: path in which to create output directory

    Outputs: new directory inside output directory named according to the above parameters, inside:
             the morphology model as a basic numpy array .txt file
             png images of the morphology model at various timesteps leading up to the target number of steps
    """
    ### Create folder to save morphology model
    counter = counter
    modPath = savePath.joinpath(f'D{D}_a{a}_eps{eps}_{nxy}pix_{int(nxy*dxy)}size_{mean}m_{steps}steps_{counter}')
    modPath.mkdir(parents=True, exist_ok=True)
    
    ### Create mesh
    mesh = Grid2D(nx=nxy, ny=nxy, dx=dxy, dy=dxy)

    ### Define volume fraction map
    phi = CellVariable(mesh=mesh, name='phi')
    phi.setValue(GaussianNoiseVariable(mesh=mesh, mean=mean, variance=0.05))

    ### Set equation 
    PHI = phi.arithmeticFaceValue
    eq = (TransientTerm()
        == DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))
        - DiffusionTerm(coeff=(D, eps**2)))
    
    ### Generate morphology:
    elapsed = 0

    # viewer = Viewer(vars=(phi,), datamin=0., datamax=1.)
    # elapsed = 0
    # viewer.cmap = cmap

    data = phi.copy().globalValue.reshape((nxy, nxy))
    np.savetxt(modPath.joinpath(f'BHJ_{elapsed}steps.txt'), data)
    plt.imshow(data, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)),
               vmin=0, vmax=1)
    plt.title(f'{nxy}x{nxy} pixels, D={D}, a={a}. epsilon={eps}, {elapsed}steps')
    plt.colorbar(label='phi')
    plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'), dpi=200)
    plt.close('all')

    time_0 = time.time()
    while elapsed < steps:
        if elapsed < 6:
            dt = 1
        elif elapsed < 30:
            dt = 2
        elif elapsed < 100:
            dt = 5
        elif elapsed < 500:
            dt = 10
        else: 
            dt = 20
        elapsed += dt  
        eq.solve(phi, dt=dt, solver=DefaultSolver(precon=None))
        if elapsed in (10, 30, 50, 100, 200, 300, 400, 500, steps):
            print(f'{elapsed} time steps in {np.round((time.time() - time_0), 1)} seconds!')
            # viewer.plot(modPath.joinpath(f'BHJ_{elapsed}steps.png'))
            data = phi.copy().globalValue.reshape((nxy, nxy))
            np.savetxt(modPath.joinpath(f'BHJ_{elapsed}steps.txt'), data)
            plt.imshow(data, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)),
                       vmin=0, vmax=1)
            plt.title(f'{nxy}x{nxy} pixels, D={D}, a={a}. epsilon={eps}, {elapsed}steps')
            plt.colorbar(label='phi')
            plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'), dpi=200)
            plt.close('all')

    print(f'Done with {nxy}pix gen!')

def gen_3d(nxy=200, nz=10, dxy=1.5, dz=1.5, mean=0.5, D=1., a=1., eps=1., steps=500, savePath=savePath, counter=1):
    """
    Generate a 3D BHJ morphology based on the Cahn-Hilliard Eqn. using a generic free energy term that penalizes 
    values of phi between 0 and 1. 

    Inputs: nxy: number of pixels in plane
            nz: number of in z
            dxy: size of each pixel in plane
            dz: size of pixel in z
            mean: starting average value for the generated 2D mesh
            D, a, eps: terms for the Cahn-Hilliard Eqn. (Diffusion Rate, free energy term, deg. of mixing term)
            steps: number of time steps to simulate
            savePath: path in which to create output directory

    Outputs: new directory inside output directory named according to the above parameters, inside:
             the morphology model as a basic numpy array .txt file
             png images of the morphology model at various timesteps leading up to the target number of steps
    """
    ### Create folder to save morphology model
    counter = counter
    modPath = savePath.joinpath(f'D{D}_a{a}_eps{eps}_{nxy}x{nz}vox_{int(nxy*dxy)}x{int(nz*dz)}size_{mean}m_{steps}steps_{counter}')
    modPath.mkdir(parents=True, exist_ok=True)
    
    ### Create mesh
    mesh = Grid3D(nx=nxy, ny=nxy, nz=nz, dx=dxy, dy=dxy, dz=dz)

    ### Define volume fraction map
    phi = CellVariable(mesh=mesh, name='phi')
    phi.setValue(GaussianNoiseVariable(mesh=mesh, mean=mean, variance=0.05))

    ### Set equation 
    PHI = phi.arithmeticFaceValue
    eq = (TransientTerm()
        == DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))
        - DiffusionTerm(coeff=(D, eps**2)))
    
    ### Generate morphology:
    elapsed = 0

    data = phi.copy().globalValue.reshape((nz, nxy, nxy))
    np.save(modPath.joinpath(f'BHJ_{elapsed}steps.npy'), data)
    fig, axs = plt.subplots(nrows=2, ncols=5)
    fig.set(tight_layout=True, size_inches=(8, 4))
    fig.suptitle(f'{nxy}x{nxy}x{nz} voxels, D={D}, a={a}. epsilon={eps}, {elapsed}steps')
    axs = axs.flatten()
    for iz in range(data.shape[0]):
        axs[iz].imshow(data[iz,:,:], vmin=0, vmax=1, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
        axs[iz].set_title(f'z={(iz+1)*dz}', fontsize=4)
        axs[iz].tick_params(axis='x', labelsize=3)
        axs[iz].tick_params(axis='y', labelsize=3)
        if iz in (0, 5):
            axs[iz].set_ylabel('[nm]', fontsize=3)
        if iz >=5:
            axs[iz].set_xlabel('[nm]', fontsize=3)
    plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'), dpi=300)
    plt.close('all')

    time_0 = time.time()
    while elapsed < steps:
        if elapsed < 6:
            dt = 1
        elif elapsed < 30:
            dt = 2
        elif elapsed < 100:
            dt = 5
        elif elapsed < 500:
            dt = 10
        else:
            dt = 20
        elapsed += dt  
        eq.solve(phi, dt=dt, solver=DefaultSolver(precon=None))
        if elapsed in (3, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, steps):
            print(f'{elapsed} time steps in {np.round((time.time() - time_0), 1)} seconds!')
            data = phi.copy().globalValue.reshape((nz, nxy, nxy))
            np.save(modPath.joinpath(f'BHJ_{elapsed}steps.npy'), data)
            fig, axs = plt.subplots(nrows=2, ncols=5)
            fig.set(tight_layout=True, size_inches=(8, 4))
            fig.suptitle(f'{nxy}x{nxy}x{nz} voxels, D={D}, a={a}. epsilon={eps}, {elapsed}steps')
            axs = axs.flatten()
            for iz in range(data.shape[0]):
                axs[iz].imshow(data[iz,:,:], vmin=0, vmax=1, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
                axs[iz].set_title(f'z={(iz+1)*dz}', fontsize=4)
                axs[iz].tick_params(axis='x', labelsize=3)
                axs[iz].tick_params(axis='y', labelsize=3)
                if iz in (0, 5):
                    axs[iz].set_ylabel('[nm]', fontsize=3)
                if iz >=5:
                    axs[iz].set_xlabel('[nm]', fontsize=3)
            plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'), dpi=300)
            plt.close('all')
    print(f'Done with {nxy}x{nz}vox gen!')

if __name__ == '__main__':
    # nxy = 632
    dxy = 1.5
    mean = 0.5
    D = a = 1.0
    eps = 1.1
    steps = 800
    counter = 1

    # gen_3d(nxy=nxy, nz=nz, dxy=dxy, dz=dz, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)

    # for eps in [1.0, 1.1]:
    #     for a in [0.5, 0.7, 1.0]:
    #         gen_2d(nxy=nxy, dxy=dxy, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)

    nxy = 134
    nz = 10
    # dz = 1.5
    for eps in [1.1]:
        for a in [0.7]:
            for dz in [2.0]:
                for counter in [1, 2, 3]:
                    gen_3d(nxy=nxy, nz=nz, dxy=dxy, dz=dz, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)

        # for eps in [1.0, 1.1]:
        #     for a in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:


    # for dxy in [1.5]:
    #     for dz in [1.5]:
    #         for eps in [1.0]:
    #             for a in [0.9]:
    #                 gen_3d(nxy=nxy, nz=nz, dxy=dxy, dz=dz, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)

    # for nxy in [632]:
    #     for counter in [1]:
    #         for eps in [1.0]:
    #             for a in [0.9]:
    #         # for eps in [1.0, 1.1]:
    #         #     for a in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #                 gen_2d(nxy=nxy, dxy=dxy, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)
