To view these instructions in JupyterLab with markdown preview, right click the file and select 'Open width'->'Markdown Preview'

# Notes to set up your conda environment (in the NSLS-II JupyterHub):
These notebooks used to work with the default NSLS-II JupyterHub python environment... not anymore. The smi_analysis and the PyHyperScattering packages both require pygix, which breaks if the silx library version is greater than 2.0.0. The default python environment in the NSLS-II JupyterHub no longer works for this import because of this (you'll get: ImportError: cannot import name 'ocl' from 'pyFAI.opencl').

So, to use these notebooks you will need to make a custom conda environment with the smi_analysis and PyHyperScattering packages. To set up a conda environment and initalize it as a kernel to use in the JupyterHub, refer to the instructions at the NSLS-II JupyterHub guide (https://jupyter.nsls2.bnl.gov/hub/guide).**

To set up your conda environment, install your conda environment (name it 'smi', or whatever other simple name you'd like). I recommend running these lines into terminal:
1. `conda create -n smi python=3.11 numpy matplotlib jupyter ipykernel ipympl dask xarray zarr rclone` (this includes most of the packages you'll need and could take a few minutes).**  
2. Activate your new conda environment: `conda activate smi`, this should update the name of the conda environment at the start of your terminal line.
3. Initalize your conda environment as a Jupyter kernel: `python -m ipykernel install --user --name smi --display-name smi` (you can list available kernels with `jupyter kernelspec list`, and you can remove one with `jupyter kernelspec uninstall smi` if ever needed)
4. Install the smi package: `pip install smi-analysis`
5. Install the PyHyperScattering package `pip install PyHyperScattering`
6. Ready to rumble!


(  ** IF you're getting a "NoWritablePkgsDirError" when creating a conda environment, type these two lines into terminal first:
1. conda config
2. conda config --add pkgs_dirs /nsls2/users/YOUR_USER_FOLDER(i.e. alevin)/.conda/pkgs  )

# Notes on data directories and copying to our proposal folder
Raw text paths for the purpose of pasting into terminal to copy/move/zip data:

Raw Paths:\
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_01` (copied as raw_01) # OPV solutions day 1 \
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_02` (copied as raw_02) # OPV films night 1\
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_03` (copied as raw_03) # OPV films day 2\
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_04` (copied as raw_04) # Polysulfide solutions day 2\
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_05` (copied as raw_05) # OPV films + Li2S static solution night 2\
`/nsls2/data/smi/legacy/results/data/2024_3/000000_Marks_01`  (copied as raw_06)  # Li2S powder final morning

Our proposal path:
`/nsls2/data/smi/proposals/2024-3/pass-316856`

Rclone copy statement to paste & run in terminal (with activated conda environment with rclone):
`rclone copy -LP /nsls2/data/smi/legacy/results/data/2024_3/000000_Chaney_0# /nsls2/data/smi/proposals/2024-3/pass-316856/raw_0#`
