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
scriptPath = pathlib.Path('/pl/active/Toney-group/anle1278/rsoxs_suite/morph_gen')
savePath = scriptPath.joinpath('testing')
cmap = plt.cm.YlGnBu_r.copy()  # Set a sequential colormap

### Define meshes
nx = ny = 100
# nz = 10
dx = dy = 1.
# dz = 1.
# mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

### Define volume fraction map
phi = CellVariable(mesh=mesh, name='phi')
phi.setValue(GaussianNoiseVariable(mesh=mesh, mean=0.5, variance=0.05))

### Set equation 
PHI = phi.arithmeticFaceValue
D = 1.
a = 1.
epsilon = 1.
eq = (TransientTerm()
      == DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))
      - DiffusionTerm(coeff=(D, epsilon**2)))

### Initialize viewer
viewer = Viewer(vars=(phi,), datamin=0., datamax=1.)
elapsed = 0
viewer.cmap = cmap

viewer.plot(filename=str(savePath.joinpath(f'viewer_{elapsed}steps.png')))


### Run morphology generation PDE EQN (computationally intensive step(s))
target = 600
time_0 = time.time()

while elapsed < target:
    if elapsed < 4:
        dt = 1
    elif elapsed < 10:
        dt = 2
    elif elapsed < 100:
        dt = 5
    elif elapsed < 500:
        dt = 10
    else: 
        dt=20
    elapsed += dt  
    eq.solve(phi, dt=dt, solver=DefaultSolver(precon=None))
    # viewer.plot()
    for elapsed_value in (100, 200, 300, 400, 500):
        if elapsed==elapsed_value:
            viewer.plot(filename=str(savePath.joinpath(f'viewer_{elapsed}steps.png')))
            print(f'{elapsed} time steps in {np.round((time.time() - time_0), 1)} seconds!')

viewer.plot(filename=str(savePath.joinpath(f'viewer_{elapsed}steps.png')))
print(f'{elapsed} time steps in {np.round((time.time() - time_0), 1)} seconds!')

