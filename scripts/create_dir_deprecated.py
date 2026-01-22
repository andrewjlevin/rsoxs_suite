"""
Output: folder for each generated morphology named based on chosen parameters:

Deprecated
"""

### Imports:
import pathlib

### Define paths:
basePath = pathlib.Path('/pl/active/Toney-group/anle1278/rsoxs_suite')
savePath = basePath.joinpath('imgs_analysis/sim_runs')

### Load morph config:

def create_dir(nxy=100, dxy=1., mean=0.5, D=1., a=1., eps=1., steps=500, savePath=savePath):
    ### Create folder to save morphology model
    directory = False
    counter = 1
    counters = []
    while directory == False:
        try: 
            modPath = savePath.joinpath(f'D{D}_a{a}_eps{eps}_{nxy}pix_{int(nxy*dxy)}size_{mean}m_{steps}steps_{counter}')
            modPath.mkdir(parents=True)
            directory = True
            counters.append(counter)
        except FileExistsError:
            counter += 1
    
    return modPath
