# Imports:
import kkcalc
import pathlib
import json
import os
from kkcalc import data
from kkcalc import kk
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy import optimize

"""Functions for reading local SST1 raw nexafs data:"""

# NEXAFS reading functions from Pete D.
# Uses timestamps in the various files to interpolate/bin signals onto energy


def read_NEXAFS(scan_num, base, detector='WAXS'):
    mesh = pd.read_csv(
        list(pathlib.Path(base, scan_num).glob('*Au Mesh*.csv'))[0])
    sample_current = pd.read_csv(
        list(pathlib.Path(base, scan_num).glob('*Sample Current*.csv'))[0])
    primary = pd.read_csv(list(base.glob(scan_num + '*primary.csv'))[0])
    beamstop = pd.read_csv(
        list(pathlib.Path(base, scan_num).glob(f'*{detector}*monitor.csv'))[0])
    return primary, mesh, sample_current, beamstop


def interp_NEXAFS(primary, mesh, sample, beamstop, detector='WAXS'):
    mesh_interp = np.interp(
        primary['time'], mesh['time'], mesh['RSoXS Au Mesh Current'])
    sample_interp = np.interp(
        primary['time'], sample['time'], sample['RSoXS Sample Current'])
    beamstop_interp = np.interp(
        primary['time'], beamstop['time'], beamstop[f'{detector} Beamstop'])
    return mesh_interp, sample_interp, beamstop_interp


def read_interp(scan_num, base, detector='WAXS'):
    """
    Inputs: scan number, pathlib directory path, detector (important!)
    Outputs the energy, mesh, sample tey, and beamstop data.
    """
    print(f'{detector} detector selected')
    primary, mesh, sample_current, beamstop = read_NEXAFS(
        scan_num, base, detector=detector)
    energy = primary['en_monoen_readback']
    mesh_interp, sample_interp, beamstop_interp = interp_NEXAFS(
        primary, mesh, sample_current, beamstop, detector=detector)
    return energy, mesh_interp, sample_interp, beamstop_interp

# Some custom functions:


def get_pol(scan_num, base):
    """
    Returns polarization and sample polarization for selected scan number in specificed path.
    This is retrieved from the baseline file.
    """
    abs_pol = round(pd.read_csv(
        list(base.joinpath(scan_num).glob('*baseline*'))[0])['en_polarization'][0], 0)
    sample_pol = round(pd.read_csv(list(base.joinpath(scan_num).glob(
        '*baseline*'))[0])['en_sample_polarization'][0], 0)
    return abs_pol, sample_pol


def get_name(scan_num, base):
    """
    Returns sample name recorded in the json file of the selected scan number/folder.
    """
    loaded_json = json.load(
        open((list(pathlib.Path(base, scan_num).glob('*json*')))[0], 'r'))
    name = loaded_json[1]['sample_name']
    return name


def list_files(path, out='names', glob='*'):
    """
    Input: pathlib filepath object
           type of output (out = 'names' or out = 'files')
           glob str selector (default is '*')
    Output: returns list of either filenames of filepaths depending on output selected
    """
    files = sorted(path.glob(glob))
    filenames = [file.name for file in files]
    if out == 'names':
        return filenames
    elif out == 'files':
        return files
    else:
        print('No output selected, choose "names" or "files"')


"""Functions for loading normalized nexafs data & estimating tilt angle:"""


def load_nexafs(fpath, samplename, angles):
    """
    Returns xarray dataset of loaded nexafs data formatted properly for rest of workflow.

    Inputs: - filepath to *.txt file with columns of energy, angle1, angle2, etc.
            - samplename: name for this nexafs sample ('Y6', 'PM6', etc)
            - angles (np array of angles in column order)
    Outputs: properly formatted xarray dataset of data
    """
    tey = np.loadtxt(fpath).T

    # wrap in an xarray
    θ_val = angles

    nexafs = xr.Dataset(
        data_vars=dict(electron_yield=(['cos_sq_θ', 'energy'], tey[1:])),
        coords={'cos_sq_θ': np.cos(θ_val * np.pi / 180) **
                2, 'θ': ('cos_sq_θ', θ_val), 'energy': tey[0]},
        attrs=dict(
            description=f'{samplename} Normalized NEXAFS', name=samplename),
    )
    nexafs.energy.attrs['unit'] = 'eV'
    nexafs['electron_yield'].attrs['unit'] = 'a.u.'
    nexafs['θ'].attrs['unit'] = '°'

    return nexafs


def show_nexafs(xmin, xmax, nexafs, exportPath, save=False, savename=None):
    """
    Plots loaded nexafs xarray dataset over specificed energy.
    Option to save figure:
        Default savename is: '{nexafs.name}_normed_nexafs_{xmin}-{xmax}.svg')
        Can be chosen by setting savename.
        Default to save into path called "exportPath", also can be chosen differently.
    """
    fig, ax = plt.subplots()
    # the xarray way of plotting
    colors = plt.cm.viridis(np.linspace(0, 0.8, nexafs.θ.size))

    for i, cos_sq_θ in enumerate(nexafs.cos_sq_θ):
        nexafs.electron_yield.sel(cos_sq_θ=cos_sq_θ).plot(
            color=colors[i], label=f'{nexafs.θ.values[i]}°', ax=ax)
    plt.title(nexafs.description)
    ax.set_xlim(xmin, xmax)
    ax.legend()
    if save == True:
        if savename == None:
            plt.savefig(exportPath.joinpath(
                f'{nexafs.name}_normed_nexafs_{xmin}-{xmax}.svg'))
        elif savename:
            plt.savefig(exportPath.joinpath(savename))
    plt.show()


def i_nexafs(α, θ):
    """
    Stöhr equation 9.16a:

    Inputs: α and θ
    Outputs: Calculated intensity value (arbitrary units)
    """
    return (1 / 3) * (1 + 0.5 * (3 * (np.cos(θ * np.pi / 180))**2 - 1) * (3 * (np.cos(α * np.pi / 180))**2 - 1))


def int_area(e_min, e_max, nexafs):
    """
    Integrates selected regions in nexafs data
    """
    return nexafs.sel(energy=slice(e_min, e_max)).integrate('energy').electron_yield.values


def mse_line_tilt(my_vars, pi_peak_areas, θ_list):
    """
    Returns mean squared error between Stöhr 9.16a and measured pi* peak areas.
    Takes difference between each point, squares them, and then sums them together.
    """
    α = my_vars[0]
    const = my_vars[1]
    return ((const * i_nexafs(α, θ_list) - pi_peak_areas)**2).sum()


def run_tilt_fit(e_min, e_max, nexafs, plot=True, savePath=None, savename=None):
    """
    Runs mse_line_tilt over selected energy region of entered nexafs xarray.

    """
    pi_peak_areas = int_area(e_min, e_max, nexafs)
    theta_list = nexafs['θ'].values

    bnds = [(0, 90), (0, 20)]
    res = optimize.differential_evolution(
        mse_line_tilt, bounds=bnds, args=(pi_peak_areas, theta_list), tol=1e-6)

    alpha, const = res.x

    if plot == True:
        # Plot intensities from Stöhr 9.16a (scaled with fitted constant) as line
        # along with measured pi_peak_areas, x-axis is cos_sq_θ:
        fig, ax = plt.subplots()
        ax.plot(nexafs.cos_sq_θ.values[:], const * i_nexafs(alpha, nexafs.θ.values[:]),
                marker='o', label=f'Stöhr 9.16a: (α={np.round(alpha,2)}, const={np.round(const,2)})',
                clip_on=False, zorder=3)
        ax.plot(nexafs.cos_sq_θ.values[:], pi_peak_areas[:], marker='o',
                label=f'{nexafs.name} NEXAFS Pi* Peak Areas', clip_on=False, zorder=4)
        ax.set(xlabel=r'$cos^2(\theta)$', ylabel='Intensity [arb. units]')
        ax.set_xticks(nexafs.cos_sq_θ.values, minor=True)
        ax.set_xlim(left=0)
        ax.legend()
        # Add secondary axis showing θ values:

        def forward(x):
            return np.arccos(np.sqrt(x)) * 180 / np.pi

        def inverse(x):
            return np.cos(x * np.pi / 180)**2

        ax2 = ax.secondary_xaxis(-0.2, functions=(forward, inverse))
        ax2.set(xlabel=r'$\theta$ ' + '$[\degree]$')
        ax2.set_xticks(nexafs.θ.values)
        
        if savePath != None:
            if savename == None:
                plt.savefig(savePath.joinpath(f'{nexafs.name}_tilt_fit_result.svg'), bbox_inches='tight', pad_inches=0.5)
            else:
                plt.savefig(savePath.joinpath(f'{savename}.svg'))

        plt.show()
            
    return res


"""Functions for running kkcalc:"""


def run_kkcalc(nexafs_xr, x_min=280, x_max=340, chemical_formula='C82H86F4N8O2S5', density=1.1):

    # save xarray as disposable text file 'scratch_nexafs.txt'
    np.savetxt('scratch_nexafs.txt',  np.c_[
               nexafs_xr.energy.values, nexafs_xr.electron_yield.values])

    # The output of kk.kk_calculate_real is f1 and f2 terms since they are calculated using Kramers-Kronig transform

    output = kk.kk_calculate_real('scratch_nexafs.txt',
                                  chemical_formula,
                                  load_options=None,
                                  input_data_type='Beta',
                                  merge_points=[x_min, x_max],
                                  add_background=False,
                                  fix_distortions=False,
                                  curve_tolerance=None,
                                  curve_recursion=100)

    # Fitting to the Henke atomic scattering factors using the given stoichiometry and formula
    stoichiometry = kk.data.ParseChemicalFormula(chemical_formula)
    formula_mass = data.calculate_FormulaMass(stoichiometry)
    ASF_E, ASF_Data = kk.data.calculate_asf(stoichiometry)
    ASF_Data2 = kk.data.coeffs_to_ASF(
        ASF_E, np.vstack((ASF_Data, ASF_Data[-1])))
    # Conversion to delta and beta and making an xarray
    n = xr.Dataset({
        'δ': (['energy'], data.convert_data(output[:, [0, 1]], 'ASF', 'refractive_index', Density=density, Formula_Mass=formula_mass)[:, 1]),
        'β': (['energy'], data.convert_data(output[:, [0, 2]], 'ASF', 'refractive_index', Density=density, Formula_Mass=formula_mass)[:, 1])},
        coords={'energy': data.convert_data(output[:, [0, 1]], 'ASF', 'refractive_index', Density=density, Formula_Mass=formula_mass)[:, 0]})

    n.energy.attrs['unit'] = 'eV'
    n.δ.attrs['unit'] = 'a.u.'
    n.β.attrs['unit'] = 'a.u.'

    os.remove('scratch_nexafs.txt')
    return n  # return an xarray


def run_kkcalc_a(nexafs_xr, x_min=280, x_max=340, chemical_formula='C82H86F4N8O2S5', density=1.1):
    n = []
    for cos_sq_θ in nexafs_xr.cos_sq_θ:
        n.append(run_kkcalc(nexafs_xr.sel(cos_sq_θ=cos_sq_θ), x_min=x_min,
                 x_max=x_max, chemical_formula=chemical_formula, density=density))
    n = xr.concat(n, dim=nexafs_xr.cos_sq_θ)
    n = n.assign_coords(θ=('cos_sq_θ', nexafs_xr.θ.values))
    n.attrs['name'] = nexafs_xr.name
    return n


def show_diel(xmin, xmax, n, save=False, savepath=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(size_inches=(8, 8))
    fig.suptitle(f'{n.name} dielectric functions')

    colors = plt.cm.viridis(np.linspace(0, 0.8, n.cos_sq_θ.size))
    colors2 = plt.cm.plasma(np.linspace(0, 0.8, n.cos_sq_θ.size))

    for i, cos_sq_θ in enumerate(n.cos_sq_θ):
        n.δ.sel(cos_sq_θ=cos_sq_θ).plot(
            ax=ax1, color=colors[i], lw=2, label=f'{n.θ[i].values}°')
        n.β.sel(cos_sq_θ=cos_sq_θ).plot(
            ax=ax2, color=colors2[i], lw=2, label=f'{n.θ[i].values}°')

    # note it is also possible to plot without an iterator using xarray's multidimensional "hue" argument as shown below, but the above allows me more control over the colors & labels.
    #n.δ.plot(ax=ax1, hue='angle')

    ymin1 = n.δ.sel(energy=slice(xmin, xmax)).min()
    ymax1 = n.δ.sel(energy=slice(xmin, xmax)).max()
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel('')
    ax1.set_ylim(ymin1 - (ymax1 - ymin1) * 0.03,
                 ymax1 + (ymax1 - ymin1) * 0.03)
    ax1.legend(loc='lower right')
    ax1.set_title('δ')

    ymax2 = n.β.sel(energy=slice(xmin, xmax)).max()

    ax2.legend(loc='upper right')
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(0, ymax2 + (ymax2) * 0.03)
    ax2.set_title('β')
    plt.setp(ax1.get_xticklabels(), visible=False)

    if save == True:
        if savepath == None:
            plt.savefig(f'{n.name}_difxns.svg')
        elif savepath:
            plt.savefig(savepath.joinpath(f'{n.name}_difxns.svg'))

    plt.show()


"""Functions for extrapolating nexafs θ and α"""


def evaluate_nexafs_fit(nexafs, nexafs_fit, new_cos_sq_θ, new_θ):
    nexafs_dummy = xr.Dataset(
        data_vars=dict(electron_yield=(['cos_sq_θ'], new_cos_sq_θ)),
        coords={'cos_sq_θ': new_cos_sq_θ},
        attrs=dict(description='Y6 NEXAFS', name=nexafs.name),
    )

    # this returns an xarray but it is unlabeled and has little of the structure,
    # labels of the original one that was fit.
    nexafs_ep = xr.polyval(nexafs_dummy.cos_sq_θ,
                           nexafs_fit.polyfit_coefficients)

    # I re-form it into a good dataset here by resorting to .values
    nexafs_ep = xr.Dataset(
        data_vars=dict(
            electron_yield=(['cos_sq_θ', 'energy'], nexafs_ep.values)),
        coords={'cos_sq_θ': new_cos_sq_θ, 'θ': (
            'cos_sq_θ', new_θ), 'energy': nexafs_ep.energy},
        attrs=dict(description=nexafs.description, name=nexafs.name),
    )
    nexafs_ep.energy.attrs['unit'] = 'eV'
    nexafs_ep['electron_yield'].attrs['unit'] = 'a.u.'
    nexafs_ep['θ'].attrs['unit'] = '°'
    return nexafs_ep
