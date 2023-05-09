import numpy as np
import xarray as xr
import os
import datetime
import time
import h5py
import cupy as cp
import cupyx.scipy.ndimage as ndigpu
#from skimage.transform import warp_polar


# Dean's function for cyrsoxs_datacubes that I used in autorun
def read_config(fname):
    config = {}
    with open(fname) as f:
        for line in f:
            key,value = line.split('=')
            key = key.strip()
            value = value.split(';')[0].strip()
            if key in ['NumX','NumY','NumZ','NumThreads','EwaldsInterpolation','WindowingType']:
                value = int(value)
            elif key in ['RotMask','WriteVTI']:
                value = bool(value)
            elif key in ['Energies']:
                value = value.replace("[", "")
                value = value.replace("]", "")
                value = np.array(value.split(","), dtype = 'float')
            else:
                value = str(value)
            config[key] = value
    return config

def warp_polar_gpu(image, center=None, radius=None, output_shape=None, **kwargs):
    """
    Function to emulate warp_polar in skimage.transform on the GPU. Not all
    parameters are supported
    
    Parameters
    ----------
    image: cupy.ndarray
        Input image. Only 2-D arrays are accepted.         
    center: tuple (row, col), optional
        Point in image that represents the center of the transformation
        (i.e., the origin in cartesian space). Values can be of type float.
        If no value is given, the center is assumed to be the center point of the image.
    radius: float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape: tuple (row, col), optional

    Returns
    -------
    polar: cupy.ndarray
        polar image
    """
    
    image = cp.asarray(image)
    if radius is None:
        radius = int(np.ceil(np.sqrt((image.shape[0] / 2)**2 + (image.shape[1] / 2)**2)))
    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    if center is not None:
        cx, cy = center
    if output_shape is None:
        output_shape = (360, radius)
    delta_theta = 360 / output_shape[0]
    delta_r = radius / output_shape[1]
    t = cp.arange(output_shape[0])
    r = cp.arange(output_shape[1])
    R, T = cp.meshgrid(r, t)
    X = R * delta_r * cp.cos(cp.deg2rad(T * delta_theta)) + cx
    Y = R * delta_r * cp.sin(cp.deg2rad(T * delta_theta)) + cy
    coordinates = cp.stack([Y, X])
    polar = cp.asnumpy(ndigpu.map_coordinates(image, coordinates, order=1))
    
    del t, r, R, T, X, Y, coordinates
    return polar

def cyrsoxs_datacubes(mydir, PhysSize, NumX, NumY):#, return_dict):
    start = datetime.datetime.now()
    #these waits are here intentionally so that data read-in can be started simultaneous with launching sim jobs and data will be read-in as soon as it is available.
    while not os.path.isfile(mydir + '/config.txt'):
        time.sleep(0.5)
    config = read_config(mydir + '/config.txt')
    
    while not os.path.isdir(mydir + '/HDF5'):
        time.sleep(0.5)
    os.chdir(mydir +'/HDF5')
    
    elist = config['Energies']
    num_energies = len(elist)
    #PhysSize = float(config['PhysSize'])
    #NumX = int(config['NumX'])
    #NumY = int(config['NumY'])
    
    #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
    hd5files = ["{:0.2f}".format(e) for e in elist]
    hd5files = np.core.defchararray.add("Energy_", hd5files)
    hd5files = np.core.defchararray.add(hd5files, ".h5")
    
    for i, e in enumerate(elist):
        while not os.path.isfile(mydir + '/HDF5/' + hd5files[i]):
            time.sleep(0.5)
        img = None
        while img is None:
            try:
                with h5py.File(hd5files[i],'r') as h5:
                    img = h5['K0']['projection'][()]
                    remeshed = warp_polar_gpu(img)
            except:
                pass
        if i==0:
            Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[1],d=PhysSize))
            Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[0],d=PhysSize))
            q = np.sqrt(Qy**2+Qx**2)
            output_chi = np.linspace(0.5,359.5,360)
            output_q = np.linspace(0,np.amax(q), remeshed.shape[1])
            data = np.zeros([NumX*NumY*num_energies])
            data_remeshed = np.zeros([len(output_chi)*len(output_q)*num_energies])
            
        data[i*NumX*NumY:(i+1)*NumX*NumY] = img[:,:].reshape(-1, order='C')
        data_remeshed[i*len(output_chi)*len(output_q):(i+1)*len(output_chi)*len(output_q)] = remeshed[:,:].reshape(-1, order='C')
    
    data = np.moveaxis(data.reshape(-1, NumY, NumX, order ='C'),0,-1)
    data_remeshed = np.moveaxis(data_remeshed.reshape(-1, len(output_chi), len(output_q), order ='C'),0,-1)
    foo = xr.DataArray(data, dims=("Qx", "Qy", "energy"), coords={"Qx":Qx, "Qy":Qy, "energy":elist})
    bar = xr.DataArray(data_remeshed, dims=("chi", "q", "energy"), coords={"chi":output_chi, "q":output_q, "energy":elist})        

    print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))
    
    return foo, bar

def datacubes_params(maindir, prefix, params):
    start = datetime.datetime.now()
    numparams = len(params)

    for j, p in enumerate(params):
    #    foo, bar = cyrsoxs_datacubes(maindir+prefix+str(p))
        mydir = maindir+prefix+str(p).zfill(4)
        if j ==0:
            #need to get all that info from config.txt including elist
            while not os.path.isfile(mydir + '/config.txt'):
                time.sleep(0.5)
            config = read_config(mydir + '/config.txt')

            while not os.path.isdir(mydir + '/HDF5'):
                time.sleep(0.5)
            os.chdir(mydir +'/HDF5')

            elist = config['Energies']
            num_energies = len(elist)
            PhysSize = float(config['PhysSize'])
            NumX = int(config['NumX'])
            NumY = int(config['NumY'])

            #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
            hd5files = ["{:0.2f}".format(e) for e in elist]
            hd5files = np.core.defchararray.add("Energy_", hd5files)
            hd5files = np.core.defchararray.add(hd5files, ".h5")
        else:
            while not os.path.isdir(mydir + '/HDF5'):
                time.sleep(0.5)
            os.chdir(mydir +'/HDF5')
        
        estart = datetime.datetime.now()
        for i, e in enumerate(elist):
            while not os.path.isfile(mydir + '/HDF5/' + hd5files[i]):
                time.sleep(0.5)
            with h5py.File(hd5files[i],'r') as h5:
                img = h5['K0']['projection']
                remeshed = warp_polar_gpu(img)
#                remeshed = warp_polar(img, order=3)
                
            if (j==0) and (i==0):
                #create all this only once
                Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[1],d=PhysSize))
                Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[0],d=PhysSize))
                q = np.sqrt(Qy**2+Qx**2)
                output_chi = np.linspace(0.5,359.5,360)
                lenchi = len(output_chi)
                output_q = np.linspace(0,np.amax(q), remeshed.shape[1])
                lenq = len(output_q)
                data = np.zeros([NumX*NumY*num_energies*numparams])
                data_remeshed = np.zeros([len(output_chi)*len(output_q)*num_energies*numparams])

            data[j*num_energies*NumX*NumY + i*NumX*NumY:j*num_energies*NumX*NumY +(i+1)*NumX*NumY] = img[:,:].reshape(-1, order='C')
            data_remeshed[j*num_energies*lenchi*lenq + i*lenchi*lenq:j*num_energies*lenchi*lenq +(i+1)*lenchi*lenq] = remeshed[:,:].reshape(-1, order='C')
        print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-estart))
    data = data.reshape(numparams*num_energies, NumY, NumX, order ='C')
    data = data.reshape(numparams, num_energies, NumY, NumX, order ='C')
    data_remeshed = data_remeshed.reshape(numparams*num_energies,lenchi, lenq, order ='C')
    data_remeshed = data_remeshed.reshape(numparams, num_energies,lenchi, lenq, order ='C')

    lfoo = xr.DataArray(data, dims=("param","energy", "Qy", "Qx"), coords={"energy":elist, "param":params, "Qy":Qy, "Qx":Qx})
    lbar = xr.DataArray(data_remeshed, dims=("param", "energy", "chi", "q"), coords={"chi":output_chi, "q":output_q, "energy":elist, "param":params})

    print(f'Finished reading ' + str(numparams) + ' parameters. Time required: ' + str(datetime.datetime.now()-start))
    #cp.cuda.Device().__exit__()
    return lfoo, lbar
