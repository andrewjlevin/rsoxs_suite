import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid3D, Viewer, GaussianNoiseVariable, Grid2D, TransientTerm, DiffusionTerm, DefaultSolver
from fipy.tools import numerix

basePath = pathlib.Path('/pl/active/Toney-group/anle1278/rsoxs_suite')
savePath = basePath.joinpath('imgs_analysis/sim_runs')
cmap = plt.cm.YlGnBu_r.copy()

def create_output_directory(save_path, params):
    counter = params.pop('counter')
    output_dir = save_path.joinpath('_'.join([f'{key}{val}' for key, val in params.items()]) + f'_{counter}')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_2d_morphology(output_dir, elapsed, nxy, dxy, data):
    np.savetxt(output_dir.joinpath(f'BHJ_{elapsed}steps.txt'), data)
    plt.imshow(data, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
    plt.colorbar(label='phi')
    plt.savefig(output_dir.joinpath(f'BHJ_{elapsed}steps.png'))
    plt.close('all')

def save_3d_morphology(output_dir, elapsed, nxy, nz, dxy, dz, data):
    np.save(output_dir.joinpath(f'BHJ_{elapsed}steps.npy'), data)
    fig, axs = plt.subplots(nrows=2, ncols=5)
    fig.set(tight_layout=True, size_inches=(8, 4))
    fig.suptitle(f'{nxy}x{nxy}x{nz} voxels, D={D}, a={a}. epsilon={eps}, {elapsed}steps')
    axs = axs.flatten()
    for iz in range(data.shape[0]):
        axs[iz].imshow(data[iz,:,:], vmin=0, vmax=1, origin='lower', cmap=cmap, extent=(0, int(nxy*dxy), 0, int(nxy*dxy)))
        axs[iz].set_title(f'z={(iz+1)*dz}', fontsize=8)
        axs[iz].tick_params(axis='x', labelsize=6)
        axs[iz].tick_params(axis='y', labelsize=6)
        if iz in (0, 5):
            axs[iz].set_ylabel('[nm]', fontsize=4)
        if iz >=5:
            axs[iz].set_xlabel('[nm]', fontsize=4)
    plt.savefig(output_dir.joinpath(f'BHJ_{elapsed}steps.png'), dpi=250)
    plt.close('all')

def gen_2d(nxy=200, dxy=1.5, mean=0.5, D=1., a=1., eps=1., steps=500, save_path=savePath, counter=1):
    params = {
        'D': D,
        'a': a,
        'eps': eps,
        'nxy': nxy,
        'dxy': dxy,
        'mean': mean,
        'steps': steps,
        'counter': counter
    }
    output_dir = create_output_directory(save_path, params)
    mesh = Grid2D(nx=nxy, ny=nxy, dx=dxy, dy=dxy)
    phi = CellVariable(mesh=mesh, name='phi')
    phi.setValue(GaussianNoiseVariable(mesh=mesh, mean=mean, variance=0.05))
    PHI
