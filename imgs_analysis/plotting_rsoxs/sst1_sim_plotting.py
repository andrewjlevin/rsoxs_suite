import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle as pkl
import pathlib
# from paramiko import SSHClient, AutoAddPolicy

# x = np.linspace(0, 2*np.pi, 1000)
# y = (np.sin(x)) * np.cos(x**3)

runPath = pathlib.Path.cwd()
userPath = runPath.joinpath('nsls2', 'users', 'alevin')

# client=SSHClient()
# client.load_host_keys(userPath.joinpath('.ssh', 'known_hosts'))
# client.load_system_host_keys()

# client.set_missing_host_key_policy(AutoAddPolicy())

def main():
    print(userPath)

if __name__ == "__main__":
    main()
