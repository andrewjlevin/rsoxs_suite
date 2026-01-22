### Imports:
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime
import dask.array
# import tifftools

## Set an RSoXS colormap for later
cm = plt.cm.terrain.copy()
cm.set_bad('purple')


### Dictionaries
# Dictionaries of starting points for SAXS & WAXS beamcenters
bcxy_2021_3 = {
    'waxs_bcx':394,
    'waxs_bcy':537  
}


bcxy_2022_2 = {
    'saxs_bcx':488,
    'saxs_bcy':515,
    'waxs_bcx':396.3,
    'waxs_bcy':553
}

### Define a sample guide (sample ID to detailed name)
sample_guide = {
    'andrew1':'PM6-Y6-CF',
    'andrew2':'PM6-Y6-CFCN',
    'andrew3':'PM6-Y6-Tol',
    'andrew4':'PM6-Y7-CF',
    'andrew5':'PM6-Y7-CFCN',
    'andrew6':'PM6-Y7-Tol',
    'andrew7':'PM6-Y7BO-CF',
    'andrew8':'PM6-Y7BO-CFCN',
    'andrew9':'PM6-Y7BO-Tol',
    'andrew10':'PM6-Y12-CF',
    'andrew11':'PM6-Y12-CFCN',
    'andrew12':'PM6-Y12-Tol',
    'andrew13':'PM7D5-Y6-CF',
    'andrew14':'PM7D5-Y6-Tol',
    'andrew15':'PM7D5-Y247-CF',
    'andrew16':'PM7D5-Y247-Tol',
    'andrew17':'PM7D5-Y12-CF',
    'andrew18':'PM7D5-Y12-CF',
    'andrew19':'PM7D5-Y12-Tol',
    'andrew20':'PM7D5-Y12-Tol'
}

# sample_guide = {
#     'Blend1':'PM7-Y6-CF',
#     'Blend2':'PM7-Y6-CFCBCN',
#     'Blend3':'PM7-Y247-CF',
#     'Blend4':'PM7-Y247-CFCBCN',
#     'Blend5':'PM7D4-Y6-CF',
#     'Blend6':'PM7D4-Y6-CFCBCN',
#     'Blend7':'PM7D5-Y6-CF',
#     'Blend8':'PM7D5-Y6-CFCBCN',
#     'Blend9':'PM7D3-Y6-CF',
#     'Blend10':'PM7D3-Y246-CF',
#     'Blend11':'PM7D3-Y247-CF',
#     'Blend12':'PM7D3-Y248-CF',
#     'Blend13':'PM7D5-Y246-CF',
#     'Blend14':'PM7D5-Y247-CF',
#     'Blend16':'PM7D5-Y12-OXY',
#     'Blend17':'PM7D5-Y12-2MeTHF',
#     'Blend18':'PM7D5-Y12-CB',
# }

# lengths = []
# for string in list(sample_guide.values()):
#     lengths.append(len(string))

# max_len = max(lengths)
# print(f'Use {max_len} spaces for blend_name while naming output folders')

# Just for smaller name for detector for labelling
detector_guide = {
    'Small Angle CCD Detector': 'SAXS',
    'Wide Angle CCD Detector': 'WAXS'
}


### Data Functions:
def dask_da_concat_pol(da00, da90):
    """
    Inputs: 2 q-space rsoxs Dask DataArrays of different polarization 
            *must have 'polarization' attribute with a scalar value inside a list*
    Outputs: Concatenated dask dataArray along a new 'pol' dimension 
    """
    da00 = da00.assign_coords({'pol': da00.polarization[0]})
    da00 = da00.expand_dims(dim={'pol':1})
    
    da90 = da90.assign_coords({'pol': da90.polarization[0]})
    da90 = da90.expand_dims(dim={'pol':1})
    
    da_out = xr.concat([da00, da90], 'pol')
    
    try:
        del da_out.attrs['polarization'], da_out.attrs['en_polarization']  # This is now a dim/coordinate
    except KeyError:
        pass
    
    return da_out


def load_stacked_pol(SST1RSoXSDB, id_p00, id_p90):
    """
    Inputs: 2 scan id's for rsoxs full carbon scans
            They should be the same detector, and two different polarizations. 
            
    Output: DataArray with extra 'pol' dimension.
    """
    da00 = SST1RSoXSDB.loadRun(id_p00)
    da90 = SST1RSoXSDB.loadRun(id_p90)
    
    da_out = dask_da_concat_pol(da00, da90)
    
    return da_out

def integrate_stacked_pol(integ, da_in):
    """
    Inputs: One raw SST1 RSoXS dask xr.DataArray with dimensions 'pol', 'energy', 'pix_x', 'pix_y'
    Outputs: One transformed RSoXS dask xr.DataArray with dimensions 'pol', 'energy', 'q', 'chi'
    """
    integ_p00 = integ.integrateImageStack_dask(da_in.sel(pol=0))
    integ_p00.attrs['polarization'] = [0]
    
    integ_p90 = integ.integrateImageStack_dask(da_in.sel(pol=90))
    integ_p90.attrs['polarization'] = [90]
    
    da_out = dask_da_concat_pol(integ_p00, integ_p90)
    da_out.attrs = da_in.attrs
    
    return da_out


def apply_q_labels(data):
    """
    Function by Pete D. for converting 'pix_x', 'pix_y' to 'qx', 'qy'
    """
    data_ds = data.to_dataset(name='images')
    data_ds['qpx'] = 2*np.pi*60e-6/(data.attrs['sdd']/1000)/((1.239842e-6/data_ds.energy)*1e10)
    data_ds['qx'] = (data_ds.pix_x-data.attrs['beamcenter_x'])*data_ds.qpx
    data_ds.qx.attrs['unit'] = '1/Å'
    data_ds['qy'] = (data_ds.pix_y-data.attrs['beamcenter_y'])*data_ds.qpx
    data_ds.qy.attrs['unit'] = '1/Å'
    data_ds_withq = data_ds.assign_coords({'qx':data_ds.qx,'qy':data_ds.qy})
    return data_ds_withq.images

### Save zarr store/directory 
def save_zarr(saxswaxs, zarrPath, prefix='raw_qxqy'):
    """
    Saves zarr stores of xr.Datasets from xr.DataArrays inputted as a list / tuple
    """
    
    for da in saxswaxs:
    ### Fix attributes to make compatible for serializing
        for k, v in da.attrs.items():
            if isinstance(v, dask.array.core.Array):
                da.attrs[k] = v.compute()
            elif isinstance(v, dict) or isinstance(v, datetime.datetime):
                da.attrs[k] = str(v)    
            # print(f'{k:<20}  |  {type(v)}')

        ### Create datasets with both detector DataArrays as variables
        ds = da.to_dataset(name=f'{da.rsoxs_config}')
        # ds.sel(energy=285, method='nearest').da.data.compute() == da.sel(energy=285, method='nearest').data.compute()

        ### Create & populate zarr stores
        ds.to_zarr(zarrPath.joinpath(f'{prefix}_{da.sample_name}_{da.blend_name}_{detector_guide[da.detector]}.zarr'), mode='w')


### Plotting functions:
def waxs_p00_p90_plot(raw_waxs, energy=285, vmax=5e3):
    """
    Plot 2 raw detector images: WAXS p00, WAXS p90
    
    Inputs: xr.DataArray with dimensions 'pol', 'pix_x', 'pix_x', 'energy'
    """
    fig, (wax00, wax90) = plt.subplots(nrows=1, ncols=2)
    fig.set(tight_layout=True, size_inches=(8,16))

    raw_waxs = raw_waxs.drop_vars('dark_id')
    raw_waxs.sel(pol=0).sel(energy=energy, method='nearest').plot.imshow(ax=wax00, origin='lower', norm=LogNorm(1e1, vmax), 
                                                                     cmap=cm, interpolation='antialiased', add_colorbar=False)
    raw_waxs.sel(pol=90).sel(energy=energy, method='nearest').plot.imshow(ax=wax90, origin='lower', norm=LogNorm(1e1, vmax), 
                                                                      cmap=cm, interpolation='antialiased', add_colorbar=False)
    
    for ax in (wax00, wax90):
        ax.set(aspect='equal')
        
    plt.show()
    plt.close('all')
    
def saxs_waxs_p00_p90_plot(raw_saxs, raw_waxs, energy=285):
    """
    Plot 4 raw detector images: SAXS p00, SAXS p90, WAXS p00, WAXS p90
    
    Inputs: raw_saxs: saxs xr.DataArray with dimensions 'pol', 'pix_x', 'pix_x', 'energy'
    """
    fig, ((sax00, sax90), (wax00, wax90)) = plt.subplots(nrows=2, ncols=2, subplot_kw=(dict()))
    fig.set(tight_layout=True, size_inches=(8,8))
    
    raw_saxs = raw_saxs.drop_vars('dark_id')    
    raw_saxs.sel(pol=0).sel(energy=energy, method='nearest').plot.imshow(ax=sax00, origin='lower', norm=LogNorm(1e1, 5e3), 
                                                                     cmap=cm, interpolation='antialiased', add_colorbar=False)
    raw_saxs.sel(pol=90).sel(energy=energy, method='nearest').plot.imshow(ax=sax90, origin='lower', norm=LogNorm(1e1, 5e3),
                                                                      cmap=cm, interpolation='antialiased', add_colorbar=False)

    raw_waxs = raw_waxs.drop_vars('dark_id')
    raw_waxs.sel(pol=0).sel(energy=energy, method='nearest').plot.imshow(ax=wax00, origin='lower', norm=LogNorm(1e1, 5e3), 
                                                                     cmap=cm, interpolation='antialiased', add_colorbar=False)
    raw_waxs.sel(pol=90).sel(energy=energy, method='nearest').plot.imshow(ax=wax90, origin='lower', norm=LogNorm(1e1, 5e3), 
                                                                      cmap=cm, interpolation='antialiased', add_colorbar=False)
    
    for ax in (sax00, sax90, wax00, wax90):
        ax.set(aspect='equal')
        
    plt.show()
    plt.close('all')


def plot_one_mask_file(draw, maskPath, sample_name, saxs_or_waxs='waxs',
                    bc_dict=bcxy_2022_2, pix_extent=250, img=None):
    """
    Plot pyhyper .json masks
    Optionally overlay masks above detector image
    
    Inputs: draw: pyhyper DrawMask object 
            sample_name: sample_name in mask filename
            saxs_img: saxs pyhyper xr.DataArray frame for a selected energy, optional
            waxs_img: waxs pyhyper xr.DataArray frame for a selected energy, optional
    """
    if saxs_or_waxs=='waxs':
        draw.load(maskPath.joinpath(f'WAXS_{sample_name}.json'))
        mask = draw.mask
        
    elif saxs_or_waxs=='saxs':
        draw.load(maskPath.joinpath(f'SAXS_{sample_name}.json'))
        mask = draw.mask
    
    else:
        print('Incorrect detector choice, please try again')
        pass


    sbx = bc_dict['saxs_bcx']
    sby = bc_dict['saxs_bcy']
    wbx = bc_dict['waxs_bcx']
    wby = bc_dict['waxs_bcy']

    fig, ax = plt.subplots()
    fig.set(tight_layout=True, size_inches=(4,4))
    
    if img is None:
        ax.imshow(mask, origin='lower')
        ax.set(title='mask', xlabel='pix_x', ylabel='pix_y')
        plt.show()
        plt.close('all')   
    
    else:
        if saxs_or_waxs=='saxs':
            img.plot.imshow(ax=ax, xlim=(sbx-pix_extent,sbx+pix_extent), ylim=(sby-pix_extent,sby+pix_extent),origin='lower', 
                                      norm=LogNorm(1e1, 5e3), cmap=cm, interpolation='antialiased', add_colorbar=False)
        else:
            img.plot.imshow(ax=ax, xlim=(wbx-pix_extent,wbx+pix_extent), ylim=(wby-pix_extent,wby+pix_extent),origin='lower', 
                                      norm=LogNorm(1e1, 5e3), cmap=cm, interpolation='antialiased', add_colorbar=False)            
        ax.imshow(mask, origin='lower', alpha=0.6)
        ax.set(title='mask', xlabel='pix_x', ylabel='pix_y')

        plt.show()
        plt.close('all')
    
def plot_mask_files(draw, maskPath, sample_name,
                    bc_dict=bcxy_2022_2, pix_extent=250, saxs_img=None, waxs_img=None):
    """
    Plot pyhyper .json masks
    Optionally overlay masks above detector image
    
    Inputs: draw: pyhyper DrawMask object 
            sample_name: sample_name in mask filename
            saxs_img: saxs pyhyper xr.DataArray frame for a selected energy, optional
            waxs_img: waxs pyhyper xr.DataArray frame for a selected energy, optional
            
    Returns: saxs_mask
             waxs_mask
    """
    draw.load(maskPath.joinpath(f'SAXS_{sample_name}.json'))
    saxs_mask = draw.mask

    draw.load(maskPath.joinpath(f'WAXS_{sample_name}.json'))
    waxs_mask = draw.mask

    sbx = bc_dict['saxs_bcx']
    sby = bc_dict['saxs_bcy']
    wbx = bc_dict['waxs_bcx']
    wby = bc_dict['waxs_bcy']

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set(tight_layout=True, size_inches=(8,4))
    
    if saxs_img is None and waxs_img is None:
        axs[0].imshow(saxs_mask, origin='lower')
        axs[0].set(title='SAXS mask', xlabel='pix_x', ylabel='pix_y')
        axs[1].imshow(waxs_mask, origin='lower')
        axs[1].set(title='WAXS mask', xlabel='pix_x', ylabel='pix_y')
        plt.show()
        plt.close('all')   
    
    else:
        saxs_img.plot.imshow(ax=axs[0], xlim=(sbx-pix_extent,sbx+pix_extent), ylim=(sby-pix_extent,sby+pix_extent),origin='lower', 
                                  norm=LogNorm(1e1, 5e3), cmap=cm, interpolation='antialiased', add_colorbar=False)
        axs[0].imshow(saxs_mask, origin='lower', alpha=0.6)
        axs[0].set(title='SAXS mask', xlabel='pix_x', ylabel='pix_y')

        waxs_img.plot.imshow(ax=axs[1], xlim=(wbx-pix_extent,wbx+pix_extent), ylim=(wby-pix_extent,wby+pix_extent),origin='lower', 
                                  norm=LogNorm(1e1, 5e3), cmap=cm, interpolation='antialiased', add_colorbar=False)
        axs[1].imshow(waxs_mask, origin='lower', alpha=0.6)
        axs[1].set(title='WAXS mask', xlabel='pix_x', ylabel='pix_y')
        plt.show()
        plt.close('all')
        
    return saxs_mask, waxs_mask
    