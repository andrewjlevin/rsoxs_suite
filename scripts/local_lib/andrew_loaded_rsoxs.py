import pathlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tifftools

## Set an RSoXS colormap
cm = plt.cm.terrain.copy()
cm.set_bad('purple')


class loaded_rsoxs:
    """
    Object to contain all noted zarr stores of raw qxqy and integ qchi dataarrays for a selected RSoxS sample
    """
    def __init__(self, sample_name: str, zarrPath: pathlib.Path):
        """
        Instantiate a loaded_rsoxs object. 
        Inputs: sample_name: This should be in the filename, located between 'raw'/'integ' and 'SAXS'/'WAXS'
                zarrPath: This is the pathlib PosixPath or WindowsPath to the directory containing the named zarr stores
        """
        
        self.sample_name = sample_name
        self.raw_saxs = xr.open_zarr(list(zarrPath.glob(f'raw*_{sample_name}_*SAXS*'))[0]).saxs
        self.raw_waxs = xr.open_zarr(list(zarrPath.glob(f'raw*_{sample_name}_*WAXS*'))[0]).waxs
        self.integ_saxs = xr.open_zarr(list(zarrPath.glob(f'integ*_{sample_name}_*SAXS*'))[0]).saxs
        self.integ_waxs = xr.open_zarr(list(zarrPath.glob(f'integ*_{sample_name}_*WAXS*'))[0]).waxs
        
        self.blend_name = self.raw_waxs.blend_name
    
    def __str__(self):
        return f"Loaded P-RSoXS Object for {self.sample_name}. Contains raw_saxs, raw_waxs, integ_saxs, and integ_waxs DataArrays"

    def build_tiff_stack(self, gif_energies, exportPath, vmax=5e3):
        """
        Build tiff stack function, needs refining but it works
        """
        for da in [self.raw_saxs, self.raw_waxs, self.integ_saxs, self.integ_waxs]:
            for pol in (0, 90):
                # Save frames as individual tiffs
                if 'pix_x' in da.dims:
                    for energy in gif_energies:
                        fig, ax = plt.subplots()
                        da.sel(pol=pol, energy=energy).plot.imshow(
                            ax=ax, x='qx', y='qy', norm=LogNorm(1e1, vmax), cmap=cm, interpolation='antialiased')
                        plt.title(f'{da.blend_name}: Pol={pol}°, Energy = {np.round(energy,2)}')
                        fig.savefig(exportPath.joinpath('tiff_frames', f'raw_{da.rsoxs_config}_pol{pol}_e{np.round(energy, 1)}.tiff'))
                        plt.close('all')

                    # Combine all frames into one tiff stack, for easy conversion to gifs in imagej
                    frame_paths = sorted(exportPath.joinpath('tiff_frames').glob(f'raw_{da.rsoxs_config}*pol{pol}*'))         
                    frame1 = tifftools.read_tiff(frame_paths[0])
                    for frame_path in frame_paths[1:]:
                        frame = tifftools.read_tiff(frame_path)
                        frame1['ifds'].extend(frame['ifds'])
                    tifftools.write_tiff(frame1, exportPath.joinpath('tiff_stacks', 
                        f'raw_qxqy_{da.sample_name}_{da.blend_name}_{da.rsoxs_config}_pol{pol}.tiff'), 
                                         allowExisting=True)   

                elif 'chi' in da.dims:
                    if da.rsoxs_config=='waxs':
                        xlim=(8e-3, 3e-1)
                    else:
                        xlim=(9e-4, 2e-2)
                    for energy in gif_energies:
                        fig, ax = plt.subplots()
                        da.sel(pol=pol, energy=energy).plot.imshow(ax=ax, xscale='log', xlim=xlim, 
                                                                   norm=LogNorm(1e1, vmax), cmap=cm, interpolation='antialiased')
                        plt.title(f'{da.blend_name}: Pol={pol}°, Energy = {np.round(energy,2)}')
                        fig.savefig(exportPath.joinpath('tiff_frames', f'integ_{da.rsoxs_config}_pol{pol}_e{np.round(energy, 1)}.tiff'))
                        plt.close('all')

                    frame_paths = sorted(exportPath.joinpath('tiff_frames').glob(f'integ_{da.rsoxs_config}*pol{pol}*'))         
                    frame1 = tifftools.read_tiff(frame_paths[0])
                    for frame_path in frame_paths[1:]:
                        frame = tifftools.read_tiff(frame_path)
                        frame1['ifds'].extend(frame['ifds'])
                    tifftools.write_tiff(frame1, exportPath.joinpath('tiff_stacks', 
                        f'integ_qchi_{da.sample_name}_{da.blend_name}_{da.rsoxs_config}_pol{pol}.tiff'), 
                                         allowExisting=True)   
    
    def chi_vals(self, saxswaxs='waxs', chi_width=30, both_sides=False):
        """
        Returns dictionary with keys being polarization and values being perp/para chi arrays
        """
        if both_sides==True:
            print('WARNING: Both sides of the detector will be included in para/perp chi wedges.' +
                  '\nThis may not be the best approach when there are large sections of masked data.')
        
        ### Load all chi values from data
        if saxswaxs=='waxs':
            chi_arr = self.integ_waxs.chi.data
        elif saxswaxs=='saxs':
            chi_arr = self.integ_saxs.chi.data   
        
        para_chi_vals = {}
        perp_chi_vals = {}
        
        for pol in (0, 90):
            ### Create 2-element arrays with the chi values corresponding to parallel/perpendicular to polarization
            para_centers = np.array([pol, pol-180])
            perp_centers = para_centers+90
            
            if pol==90:
                if saxswaxs=='waxs':
                    ### Top & Left Wedges
                    para_chi_vals[pol] = chi_arr[(chi_arr > para_centers[0]-(chi_width/2)) * (chi_arr < para_centers[0]+(chi_width/2))]
                    perp_chi_vals[pol] = np.append(chi_arr[chi_arr < -perp_centers[0]+(chi_width/2)], chi_arr[chi_arr > perp_centers[0]-(chi_width/2)])
                elif saxswaxs=='saxs':
                    ### Bottom and Left Wedge
                    para_chi_vals[pol] = chi_arr[(chi_arr > para_centers[1]-(chi_width/2)) * (chi_arr < para_centers[1]+(chi_width/2))]
                    perp_chi_vals[pol] = np.append(chi_arr[chi_arr < -perp_centers[0]+(chi_width/2)], chi_arr[chi_arr > perp_centers[0]-(chi_width/2)])
                    
                if both_sides==True:
                    para_chi_vals[pol] = np.append(para_chi_vals[pol], chi_arr[(chi_arr > para_centers[1]-(chi_width/2)) * (chi_arr < para_centers[1]+(chi_width/2))])
                    perp_chi_vals[pol] = np.append(perp_chi_vals[pol], chi_arr[(chi_arr > perp_centers[1]-(chi_width/2)) * (chi_arr < perp_centers[1]+(chi_width/2))])
            
            elif pol==0:
                if saxswaxs=='waxs':
                    ### Top & Left Wedges
                    perp_chi_vals[pol] = chi_arr[(chi_arr > perp_centers[0]-(chi_width/2)) * (chi_arr < perp_centers[0]+(chi_width/2))]
                    para_chi_vals[pol] = np.append(chi_arr[chi_arr < para_centers[1]+(chi_width/2)], chi_arr[chi_arr > -para_centers[1]-(chi_width/2)])
                elif saxswaxs=='saxs':
                    ### Bottom and Left Wedge
                    perp_chi_vals[pol] = chi_arr[(chi_arr > perp_centers[1]-(chi_width/2)) * (chi_arr < perp_centers[1]+(chi_width/2))]
                    para_chi_vals[pol] = np.append(chi_arr[chi_arr < para_centers[1]+(chi_width/2)], chi_arr[chi_arr > -para_centers[1]-(chi_width/2)])
                
                if both_sides==True: 
                    perp_chi_vals[pol] = np.append(perp_chi_vals[pol], chi_arr[(chi_arr > perp_centers[1]-(chi_width/2)) * (chi_arr < perp_centers[1]+(chi_width/2))])
                    para_chi_vals[pol] = np.append(para_chi_vals[pol], chi_arr[(chi_arr > para_centers[0]-(chi_width/2)) * (chi_arr < para_centers[0]+(chi_width/2))])
            
        return para_chi_vals, perp_chi_vals

    def isi(self, chi_width=30, saxswaxs='waxs', qrange=(1.5e-2, 8e-1), both_sides=False):
        """
        Calculate integrated scattering intensity for chosen regions in q-chi space: chi-integrated, multiplied by q^2, q-integrated
        Inputs: chi_width, total width of slice for looking at ISI for para/perp regions
        Returns: para_isi_pol: dictionary with polarization as the key, para ISI DataArrays
                 perp_isi_pol: dictionary with polarization as the key, perp ISI DataArrays
                 para_isi_avg: para ISI DataArray, averaged polarizations
                 perp_isi_avg: perp ISI DataArray, averaged polarizations
        """
        ### Load dictionaries of chi values depending on chi_width:
        para_chi_vals, perp_chi_vals = self.chi_vals(saxswaxs=saxswaxs, chi_width=chi_width, both_sides=both_sides)
        
        ### Initialize a couple dictionaries for para & perp ISI DataArrays
        para_isi_pol = {}
        perp_isi_pol = {}

        for pol in (0, 90):
            ### Perform chi average, multiply by q^2, perform q integration
            if saxswaxs=='waxs':
                perp_isi_pol[pol] = ((self.integ_waxs.sel(pol=pol, 
                    chi=perp_chi_vals[pol]).mean('chi').sel(q=slice(qrange[0], qrange[1])) * self.integ_waxs.q.sel(q=slice(qrange[0], qrange[1]))**2).sum('q'))
                para_isi_pol[pol] = ((self.integ_waxs.sel(pol=pol, 
                    chi=para_chi_vals[pol]).mean('chi').sel(q=slice(qrange[0], qrange[1])) * self.integ_waxs.q.sel(q=slice(qrange[0], qrange[1]))**2).sum('q'))
            elif saxswaxs=='saxs':
                perp_isi_pol[pol] = ((self.integ_saxs.sel(pol=pol, 
                    chi=perp_chi_vals[pol]).mean('chi').sel(q=slice(qrange[0], qrange[1])) * self.integ_saxs.q.sel(q=slice(qrange[0], qrange[1]))**2).sum('q'))
                para_isi_pol[pol] = ((self.integ_saxs.sel(pol=pol, 
                    chi=para_chi_vals[pol]).mean('chi').sel(q=slice(qrange[0], qrange[1])) * self.integ_saxs.q.sel(q=slice(qrange[0], qrange[1]))**2).sum('q'))

        ### Average calculated isi's for both polarizations
        perp_isi_avg = (perp_isi_pol[0] + perp_isi_pol[90])/2
        para_isi_avg = (para_isi_pol[0] + para_isi_pol[90])/2

        return para_isi_pol, perp_isi_pol, para_isi_avg, perp_isi_avg
          
    def e_map(self, saxswaxs='waxs', axis_meaned='chi', axis_lim=True, chi_width=30, qrange=(1.5e-2, 8e-1), both_sides=False):
        """
        For a selected qrange, take the mean over either para/perp chi slices or q to plot EvsQ or EvsChi maps, respectively
        Inputs: saxswaxs: 'waxs' or 'saxs', 
                axis_meaned: 'chi' or 'q', 
                chi_width: int (default 30) only applicable to chi-meaned 
                qrange: (num, num) default good for waxs
                both_sides: whether to use both sides of polarization images for chi wedges... probably not the best
        Returns: para_EvsQ_pol: dictionary with polarization as the key, para EvsQ DataArrays
                 perp_EvsQ_pol: dictionary with polarization as the key, perp EvsQ DataArrays
                 para_EvsQ_avg: para EvsQ DataArray, averaged polarizations
                 perp_EvsQ_avg: perp EvsQ DataArray, averaged polarizations
                 
                 OR
                 
                 qmeaned_EvsQ_pol: dictionary with polarization as the key, qmeaned EvsQ DataArrays
        """
        ### Load dictionaries of chi values depending on chi_width:
        para_chi_vals, perp_chi_vals = self.chi_vals(saxswaxs=saxswaxs, chi_width=chi_width, both_sides=both_sides)
        
        if axis_meaned=='chi':
            ### Initialize dictionaries for DataArrays
            para_EvsQ_pol = {}
            perp_EvsQ_pol = {}

            for pol in (0, 90):             
                ### Appropriately slice & mean selected dataarray 
                if saxswaxs=='waxs':
                    perp_EvsQ_pol[pol] = self.integ_waxs.sel(pol=pol, chi=perp_chi_vals[pol], q=slice(qrange[0], qrange[1])).mean('chi')
                    para_EvsQ_pol[pol] = self.integ_waxs.sel(pol=pol, chi=para_chi_vals[pol], q=slice(qrange[0], qrange[1])).mean('chi')
                elif saxswaxs=='saxs':
                    perp_EvsQ_pol[pol] = self.integ_saxs.sel(pol=pol, chi=perp_chi_vals[pol], q=slice(qrange[0], qrange[1])).mean('chi')
                    para_EvsQ_pol[pol] = self.integ_saxs.sel(pol=pol, chi=para_chi_vals[pol], q=slice(qrange[0], qrange[1])).mean('chi')

            ### Average calculated EvsQs for both polarizations
            perp_EvsQ_avg = (perp_EvsQ_pol[0] + perp_EvsQ_pol[90])/2
            para_EvsQ_avg = (para_EvsQ_pol[0] + para_EvsQ_pol[90])/2

            return para_EvsQ_pol, perp_EvsQ_pol, para_EvsQ_avg, perp_EvsQ_avg

        elif axis_meaned=='q':
            ### Initialize dictionaries for DataArrays
            qmeaned_EvsQ_pol = {}
            
            for pol in (0, 90):             
                ### Perform chi integration, multiply by q^2, perform q integration
                if saxswaxs=='waxs':            
                    qmeaned_EvsQ_pol[pol] = self.integ_waxs.sel(pol=pol, q=slice(qrange[0], qrange[1])).mean('q')
                elif saxswaxs=='saxs':
                    qmeaned_EvsQ_pol[pol] = self.integ_saxs.sel(pol=pol, q=slice(qrange[0], qrange[1])).mean('q')
            
            return qmeaned_EvsQ_pol 

    def ar_map(self, saxswaxs='waxs', axis_meaned='chi', axis_lim=True, chi_width=30, qrange=(1.5e-2, 8e-1), both_sides=False):
        """
        For a selected qrange, take the mean over either para/perp chi slices or q to plot EvsQ or EvsChi maps, respectively
        Inputs: saxswaxs: 'waxs' or 'saxs', 
                chi_width: int (default 30)
                qrange: (num, num) default good for waxs
                both_sides: whether to use both sides of polarization images for chi wedges... probably not the best
        Returns: ARvsQ_pol: dictionary containing AR dataarrays for different polarizations
                 ARvsQ_avg: AR averaged for both polarizations
        """
        ### Load dictionaries of chi values depending on chi_width:
        para_chi_vals, perp_chi_vals = self.chi_vals(saxswaxs=saxswaxs, chi_width=chi_width, both_sides=both_sides)

        ### Load para/perp EvsQ maps
        para_EvsQ_pol, perp_EvsQ_pol, a, b = self.e_map(saxswaxs=saxswaxs, chi_width=chi_width, qrange=qrange, both_sides=both_sides)
        
        ### Initialize dictionaries for DataArrays
        ARvsQ_pol = {}
        
        for pol in (0, 90):
            ARvsQ_pol[pol] = (para_EvsQ_pol[pol]-perp_EvsQ_pol[pol])/(para_EvsQ_pol[pol]+perp_EvsQ_pol[pol])
        
        ARvsQ_avg = (ARvsQ_pol[0]+ARvsQ_pol[90])/2
        
        return ARvsQ_pol, ARvsQ_avg
