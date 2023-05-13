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


def gen_2d(nxy=100, dxy=1., mean=0.5, D=1., a=1., eps=1., steps=500, savePath=savePath, counter=1):
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
    plt.imshow(data, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
    plt.colorbar(label='phi')
    plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'))
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
            plt.imshow(data, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
            plt.colorbar(label='phi')
            plt.savefig(modPath.joinpath(f'BHJ_{elapsed}steps.png'))
            plt.close('all')

    print(f'Done with {nxy}pix gen!')

if __name__ == '__main__':
    # nxy = 420
    dxy = 1.5
    mean = 0.5
    D = a = eps = 1.0
    steps = 500

    for nxy in [632]:
        for counter in [2]:
            gen_2d(nxy=nxy, dxy=dxy, mean=mean, D=D, a=a, eps=eps, steps=steps, counter=counter)
