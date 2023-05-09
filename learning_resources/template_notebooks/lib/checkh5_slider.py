# NEW CheckH5 with Slider Widgets!
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc, gridspec
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import h5py
import datetime
import warnings

# Slider Widget
%matplotlib ipympl

def components_slider_4d(single_4d_input, mask1=[], mask2=[], color='Greys', title='', cbarname = ''):
    """
    Display 3d ndarrays with a slider to move along the third dimension.
    Multiple numpy.ndarray objects can be inputted as *args. 
    Vertical slider traverses ndarray objects (components).
    Bottom slider changes dimension of view (z, y, x).
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # Sets default axis as top down view
    axis = 0
    
    # Makes 4d input into array of 3d inputs to work with previous version of components_slider
    args = np.array([single_4d_input[i,...] for i in range(single_4d_input.shape[0])])
    
    # Number of components
    comp = len(args)
    
    # Raising error if inputs are not 3d ndarrays
    for n in range(comp):
        try:
            if not args[n].ndim == 3:
                raise ValueError("Input data should be an ndarray with ndim == 3")
        except AttributeError:
              raise ValueError("Input data should be an ndarray with ndim == 3")
        except:
              raise ValueError("Input data should be an ndarray with ndim == 3")

    # Figure generation
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.05, bottom=0.28)

    # Select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = args[0][tuple(s)].squeeze()
    
    # Normalization of material colors for plotting
    if color == 'Greys':
        l = ax.imshow(im,origin='lower', cmap=color, vmin=0, vmax=1,interpolation='none')
    else:
        m1_args = np.array([mask1[i,...] for i in range(mask1.shape[0])])
        m2_args = np.array([mask2[i,...] for i in range(mask2.shape[0])])
        mask1_im = m1_args[0][tuple(s)].squeeze()
        mask2_im = m2_args[0][tuple(s)].squeeze()
        l = ax.imshow(np.ma.masked_array(im, np.logical_or(mask1_im<0.01, mask2_im<0.01)), origin='lower', cmap=color, vmin=0, vmax=np.pi,interpolation='none')
    
    fig.suptitle('            '+title, fontsize=16)
    ax.set_xlabel('X index', fontsize=12)
    ax.set_ylabel('Y index', fontsize=12)
    
    Vfrac_cbar = plt.colorbar(l, fraction=0.040)
    Vfrac_cbar.set_label(cbarname, rotation=270, labelpad = 14, fontsize=12)
    
    # Z = 1 in model
    if args[0].shape[0] != 1:
        # Horizontal slider to move through 3rd dimension
        ax0 = fig.add_axes([0.25, 0.1, 0.65, 0.03],frameon=True)
        max_dim = args[0].shape[axis] - 1
        axis_slider = Slider(ax0, 'Axis index', 0, args[0].shape[axis] - 1, valinit=0, valstep=1) #valfmt='%i'
    
        # Horizontal slider to move through different view
        ax3 = fig.add_axes([0.25, 0.01, 0.65, 0.03],frameon=True)
        view_slider = Slider(ax3, 'View', 0, 2, valinit=0, valstep=1) #valfmt='%i'
    
    # Make a vertically oriented slider to control the material 
    ax2 = fig.add_axes([0.1, 0.25, 0.0225, 0.63],frameon=True)
    mat_slider = Slider(ax2, 'Material', 1, len(args), valinit=0, orientation="vertical", valstep=1)

    def update(val):
        ind2 = int(mat_slider.val)-1
        ind = 0
        ind3 = 0
        if args[0].shape[0] != 1:
            ind = int(axis_slider.val)
            ind3 = int(view_slider.val)
            if ind3 == 0:
                ax.set_xlabel('X index', fontsize=12)
                ax.set_ylabel('Y index', fontsize=12)
                fig.subplots_adjust(left=0.05, bottom=0.28)
                Vfrac_cbar.set_label(cbarname, rotation=270, labelpad = 14, fontsize=12)
            elif ind3 == 1:
                ax.set_xlabel('X index', fontsize=12)
                ax.set_ylabel('Z index', fontsize=12)
                fig.subplots_adjust(left=0.25, bottom=0.28)
                Vfrac_cbar.set_label(cbarname, rotation=270, labelpad = 10, fontsize=10)
            elif ind3 == 2:
                ax.set_xlabel('Y index', fontsize=12)
                ax.set_ylabel('Z index', fontsize=12)
                fig.subplots_adjust(left=0.25, bottom=0.28)
                Vfrac_cbar.set_label(cbarname, rotation=270, labelpad = 10, fontsize=10)

        s = [slice(ind, ind + 1) if i == ind3 else slice(None) for i in range(3)]
        #im = args[ind2][tuple(s)].squeeze()
        if color == 'Greys':
            im = args[ind2][tuple(s)].squeeze()
            l = ax.imshow(im,origin='lower', cmap=color, vmin=0, vmax=1,interpolation='none')
        else:
            mask1_im = m1_args[ind2][tuple(s)].squeeze()
            mask2_im = m2_args[ind2][tuple(s)].squeeze()
            im = np.ma.masked_array(args[ind2][tuple(s)].squeeze(), np.logical_or(mask1_im<0.01, mask2_im<0.01))
            if im.mask.all() == True:
                im = args[ind2][tuple(s)].squeeze()
                l = ax.imshow(im+1,origin='lower', cmap='Greys', vmin=0, vmax=np.pi,interpolation='none')
            else:
                l = ax.imshow(im, origin='lower', cmap=color, vmin=0, vmax=np.pi,interpolation='none')

        l.set_data(im)
        if args[0].shape[0] != 1:
            axis_slider.ax.set_xlim(0,args[0].shape[ind3] - 1)
            axis_slider.valmax = args[0].shape[ind3] - 1
            
        update_range(val)
            
        fig.canvas.draw_idle()

    mat_slider.on_changed(update)
    
    if args[0].shape[0] != 1:
        axis_slider.on_changed(update)
        view_slider.on_changed(update)

    plt.show() 


def check_NumMat(f, morphology_type):
    morphology_num = f['Morphology_Parameters/NumMaterial'][()]
    
    if morphology_type == 0:
        num_mat = 0
        while f'Euler_Angles/Mat_{num_mat + 1}_Vfrac' in f.keys():
            num_mat +=1
    elif morphology_type == 1:
        num_mat = 0
        while f'Vector_Morphology/Mat_{num_mat + 1}_unaligned' in f.keys():
            num_mat +=1
    
    assert morphology_num==num_mat, 'Number of materials does not match manual count of materials. Recheck hdf5'
    
    return num_mat
    
    

def checkH5_vector(filename):
    # read in vector morphology
    
    with h5py.File(filename,'r') as f:
        num_mat = check_NumMat(f,morphology_type=1)
        
        ds = f['Vector_Morphology/Mat_1_unaligned'][()]
        
        unaligned = np.zeros((num_mat,*ds.shape))
        alignment = np.zeros((num_mat,*ds.shape,3))
        
        for i in range(0, num_mat):
            unaligned[i,...] = f[f'Vector_Morphology/Mat_{i+1}_unaligned'][()]
            alignment[i,...] = f[f'Vector_Morphology/Mat_{i+1}_alignment'][()]
        
        # calculate total material
        total_material = np.sum(unaligned,axis=0) + np.sum(alignment**2,axis=(0,-1))
    
    # assert that the entire morphology has total material equal to 1
    assert np.allclose(total_material,1), 'Not all voxels in morphology have Total Material equal to 1'
    
    # convert vector to Euler for visualization purposes
    S = np.zeros((num_mat, *ds.shape))
    Vfrac = S.copy()
    theta = S.copy()
    psi = S.copy()
    
    S = np.sum(alignment**2,axis=-1)
    Vfrac = unaligned + S
    
    #calculate theta and psi from vectors
    z = np.array([0,0,1])
    x = np.array([1,0,0])
    with np.errstate(invalid='ignore'):
        normed_vectors = alignment/np.sqrt(S[:,:,:,:,np.newaxis])
    np.nan_to_num(normed_vectors,copy=False)
    theta = np.arccos(np.dot(normed_vectors,z))
    psi = np.arccos(np.dot(normed_vectors,x))
    
    return Vfrac, S, theta, psi


def checkH5_euler(filename):
    # read in vector morphology
    with h5py.File(filename,'r') as f:
        num_mat = check_NumMat(f,morphology_type=0)
        
        ds = f['Euler_Angles/Mat_1_Vfrac'][()]
        
        Vfrac = np.zeros((num_mat,*ds.shape))
        S = Vfrac.copy()
        theta = Vfrac.copy()
        psi = Vfrac.copy()

        #'Mat_1_Psi', 'Mat_1_S', 'Mat_1_Theta', 'Mat_1_Vfrac'

        for i in range(0, num_mat):
            Vfrac[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Vfrac']
            S[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_S']
            theta[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Theta']
            psi[i,:,:,:] = f[f'Euler_Angles/Mat_{i+1}_Psi']
            psi = np.mod(psi, np.pi)
            # calculate total material
            total_material = np.sum(Vfrac,axis=0)
    
    # assert that the entire morphology has total material equal to 1
    assert np.allclose(total_material,1), 'Not all voxels in morphology have Total Material equal to 1' 
    
    
    return Vfrac, S, theta, psi

def checkH5_slider(filename='perp82.hd5', subsample = None, outputmat = None, runquiet = False, plotstyle='light'):
    
    #Can plot with light or dark background
    style_dict = {'dark':'dark_background',
                  'light':'default'}
    
    begin_time = datetime.datetime.now()
    
    with h5py.File(filename, 'r') as f:
        # check morphology type
        if 'Euler_Angles' in f.keys():
            morphology_type = 0
        elif 'Vector_Morphology' in f.keys():
            morphology_type = 1
        else:
            warnings.warn('Neither \"Euler_Angles\" or \"Vector_Morphology\" group detected in hdf5')
            morphology_type = None
    
    if morphology_type == 0:
        Vfrac, S, theta, psi = checkH5_euler(filename)
        num_mat, zdim, ydim, xdim = Vfrac.shape
    elif morphology_type == 1:
        Vfrac, S, theta, psi = checkH5_vector(filename)
        num_mat, zdim, ydim, xdim = Vfrac.shape
        
    
    
    
    if subsample is None:
        subsample = ydim # y dimension
        
    if not runquiet:
        print(f'Dataset dimensions (Z,Y,X): {zdim} × {ydim} × {xdim}')
        print(f'Number of Materials: {num_mat}')
        print(f'Total Vfrac whole model. Min: {np.amin(np.sum(Vfrac, axis=0))} Max: {np.amax(np.sum(Vfrac, axis=0))}')
        print('')

        
    if not runquiet:
        
        plt.style.use(style_dict[plotstyle])
        font = {'family' : 'sans-serif',
                'sans-serif' : 'DejaVu Sans',
                'weight' : 'regular',
                'size'   : 18}

        rc('font', **font)
        for i in range(0,num_mat):    
            print(f'Material {i+1} Vfrac. Min: {np.amin(Vfrac[i,:,:,:])} Max: {np.amax(Vfrac[i,:,:,:])}')
            print(f'Material {i+1} S. Min: {np.amin(S[i,:,:,:])} Max: {np.amax(S[i,:,:,:])}')
            print(f'Material {i+1} theta. Min: {np.amin(theta[i,:,:,:])} Max: {np.amax(theta[i,:,:,:])}')
            print(f'Material {i+1} psi. Min: {np.amin(psi[i,:,:,:])} Max: {np.amax(psi[i,:,:,:])}')
            print('\n')



        start = int(ydim/2)-int(subsample/2)
        end = int(ydim/2)+int(subsample/2)

        components_slider_4d(Vfrac[start:end, start:end],color='Greys', title='Vfrac',cbarname='Vfrac: volume fraction')
    
        components_slider_4d(S[start:end, start:end],color='Greys', title='S', cbarname='S: orientational order parameter')
        
        components_slider_4d(theta[start:end, start:end], mask1=Vfrac[start:end, start:end], mask2=S[start:end, start:end] ,color='jet', title='theta', cbarname='radians')
        
        components_slider_4d(psi[start:end, start:end], mask1=Vfrac[start:end, start:end], mask2=S[start:end, start:end],color='hsv', title='psi', cbarname='radians')
            
    plt.rcParams.update(plt.rcParamsDefault)

    if not runquiet:
        print(datetime.datetime.now() - begin_time)
    if not (outputmat is None):
        start = int(ydim/2)-int(subsample/2)
        end = int(ydim/2)+int(subsample/2)
        return Vfrac[outputmat,start:end,start:end], S[outputmat,start:end,start:end], psi[outputmat,start:end,start:end], theta[outputmat,start:end,start:end]
