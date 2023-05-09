import sys
sys.path.insert(1, '/home/ceb10/Jan2023/')
from dean_reduce06 import *

lfoo, lbar = cyrsoxs_datacubes('/home/ceb10/Jan2023//', 1.95, 2048, 2048)

print(lbar)
lbar.to_netcdf('/home/ceb10/Jan2023//tempS0p5.nc')
