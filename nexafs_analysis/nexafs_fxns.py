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


"""Functions for loading normalized nexafs data & estimating tilt angle:"""

def i_nexafs(alpha, theta):
    """
    Stöhr equation 9.16a:

    Inputs: alpha and theta
    Outputs: Calculated intensity value (arbitrary units)
    """
    return (1 / 3) * (1 + 0.5 * (3 * (np.cos(theta * np.pi / 180))**2 - 1) * (3 * (np.cos(alpha * np.pi / 180))**2 - 1))


def int_area(e_min, e_max, nf_DA):
    """
    Integrates selected regions in nexafs data
    """
    return nf_DA.sel(energy=slice(e_min, e_max)).integrate('energy').values


def mse_line_tilt(my_vars, pi_peak_areas, theta_list):
    """
    Returns mean squared error between Stöhr 9.16a and measured pi* peak areas.
    Takes difference between each point, squares them, and then sums them together.
    """
    alpha = my_vars[0]
    const = my_vars[1]
    return ((const * i_nexafs(alpha, theta_list) - pi_peak_areas)**2).sum()


def run_tilt_fit(e_min, e_max, nf_DA, plot=True, savePath=None, savename=None):
    """
    Runs mse_line_tilt over selected energy region of entered nexafs xarray.

    """
    pi_peak_areas = int_area(e_min, e_max, nf_DA)
    theta_list = nf_DA['theta'].values

    bnds = [(0, 90), (0, 20)]
    res = optimize.differential_evolution(
        mse_line_tilt, bounds=bnds, args=(pi_peak_areas, theta_list), tol=1e-6)

    alpha, const = res.x

    if plot == True:
        # Plot intensities from Stöhr 9.16a (scaled with fitted constant) as line
        # along with measured pi_peak_areas, x-axis is cos_sq_theta:
        fig, axs = plt.subplots(ncols=2, figsize=(15,3), dpi=120)
        fig.suptitle(str(nf_DA.sample_name.values), y=1.03, fontsize=14)
        axs[0].plot(nf_DA.cos_sq_theta.values[:], const * i_nexafs(alpha, nf_DA.theta.values[:]),
                marker='o', label=f'Stöhr 9.16a: ($\\alpha$={np.round(alpha,2)}, const={np.round(const,2)})',
                clip_on=False, zorder=3)
        axs[0].plot(nf_DA.cos_sq_theta.values[:], pi_peak_areas[:], marker='o',
                label=f'NEXAFS integrated areas', clip_on=False, zorder=4)
        axs[0].set(title='Peak fit', xlabel=r'$cos^2(\theta)$', ylabel='Intensity [arb. units]')
        axs[0].set_xticks(nf_DA.cos_sq_theta.values, minor=True)
        axs[0].set_xlim(left=0)
        axs[0].legend(loc='upper left')
        
        # Add secondary axis showing theta values:
        def forward(x):
            return np.arccos(np.sqrt(x)) * 180 / np.pi

        def inverse(x):
            return np.cos(x * np.pi / 180)**2

        ax2 = axs[0].secondary_xaxis(-0.23, functions=(forward, inverse))
        ax2.set(xlabel=r'$\theta$ [$\degree$]')
        ax2.set_xticks(nf_DA.theta.values)
        
        
        colors = plt.get_cmap('plasma_r')(np.linspace(0.15,1,len(nf_DA.theta)))
        # fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        for i, theta_val in enumerate(nf_DA.theta.values):
            (nf_DA.sel(theta=theta_val, energy=slice(281, 293))
                   .plot.line(ax=axs[1], color=colors[i], label=f'{int(theta_val)}°'))

        axs[1].axvline(e_min, color='grey')
        axs[1].axvline(e_max, color='grey')
        axs[1].set(title='NEXAFS', xlabel='X-ray Energy [eV]', ylabel='Normalized NEXAFS [arb. units]')
        axs[1].legend(title=r'$\theta$ [$\degree$]', loc='upper left')
        
        if savePath is not None and savename is not None:
            plt.savefig(savePath.joinpath(f'{savename}.png'), dpi=120)

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
    for cos_sq_theta in nexafs_xr.cos_sq_theta:
        n.append(run_kkcalc(nexafs_xr.sel(cos_sq_theta=cos_sq_theta), x_min=x_min,
                 x_max=x_max, chemical_formula=chemical_formula, density=density))
    n = xr.concat(n, dim=nexafs_xr.cos_sq_theta)
    n = n.assign_coords(theta=('cos_sq_theta', nexafs_xr.theta.values))
    n.attrs['name'] = nexafs_xr.name
    return n


def show_diel(xmin, xmax, n, save=False, savepath=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(size_inches=(8, 8))
    fig.suptitle(f'{n.name} dielectric functions')

    colors = plt.cm.viridis(np.linspace(0, 0.8, n.cos_sq_theta.size))
    colors2 = plt.cm.plasma(np.linspace(0, 0.8, n.cos_sq_theta.size))

    for i, cos_sq_theta in enumerate(n.cos_sq_theta):
        n.δ.sel(cos_sq_theta=cos_sq_theta).plot(
            ax=ax1, color=colors[i], lw=2, label=f'{n.theta[i].values}°')
        n.β.sel(cos_sq_theta=cos_sq_theta).plot(
            ax=ax2, color=colors2[i], lw=2, label=f'{n.theta[i].values}°')

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


"""Functions for extrapolating nexafs theta and alpha"""


def evaluate_nexafs_fit(nexafs, nexafs_fit, new_cos_sq_theta, new_theta):
    nexafs_dummy = xr.Dataset(
        data_vars=dict(electron_yield=(['cos_sq_theta'], new_cos_sq_theta)),
        coords={'cos_sq_theta': new_cos_sq_theta},
        attrs=dict(description='Y6 NEXAFS', name=nexafs.name),
    )

    # this returns an xarray but it is unlabeled and has little of the structure,
    # labels of the original one that was fit.
    nexafs_ep = xr.polyval(nexafs_dummy.cos_sq_theta,
                           nexafs_fit.polyfit_coefficients)

    # I re-form it into a good dataset here by resorting to .values
    nexafs_ep = xr.Dataset(
        data_vars=dict(
            electron_yield=(['cos_sq_theta', 'energy'], nexafs_ep.values)),
        coords={'cos_sq_theta': new_cos_sq_theta, 'theta': (
            'cos_sq_theta', new_theta), 'energy': nexafs_ep.energy},
        attrs=dict(description=nexafs.description, name=nexafs.name),
    )
    nexafs_ep.energy.attrs['unit'] = 'eV'
    nexafs_ep['electron_yield'].attrs['unit'] = 'a.u.'
    nexafs_ep['theta'].attrs['unit'] = '°'
    return nexafs_ep
