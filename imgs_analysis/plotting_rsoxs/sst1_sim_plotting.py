import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle as pkl
import pathlib

x = np.linspace(0, 2*np.pi, 1000)
y = (np.sin(x)) * np.cos(x**3)

def main():
    plt.plot(x, y)
    plt.savefig('test.png')
    plt.show()
    plt.close('all')

if __name__ == "__main__":
    main()
