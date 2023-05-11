""" 
Slicing SST1 Experimental & CyRSoXS Simulation Data 
"""

### Imports
import PyHyperScattering as phs
import pathlib
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# print(f'Using PyHyperScattering Version: {phs.__version__}')

### Define pathlib paths: zarr datasets & sim datasets & custom scripts
rootPath = pathlib.Path('/pl/active/Toney-group/anle1278/rsoxs_suite')
zarrPath = rootPath.joinpath('imgs_analysis/zarr_datasets_Jul2022_v1')
simsPath = rootPath.joinpath('imgs_analysis/runs')
libPath = rootPath.joinpath('imgs_analysis/plotting_rsoxs/local_lib')
sys.path.append(str(libPath))
from andrew_rsoxs_fxns import *  # type: ignore
from andrew_loaded_rsoxs import *  # type: ignore

### Set an RSoXS colormap for later
cmap = plt.cm.terrain.copy()
cmap.set_bad('purple')

### Functions

def main():
    pass

if __name__ == "__main__":
    main()
