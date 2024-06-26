import numpy as np
import os
import numexpr as ne
from scipy.stats import norm
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist


def sq(pos,qs):
    '''
    Calculates the scattering profile using the debye equation. 

    Input
      pos = scatterer positions in 3D cartesian coordinates (nx3 array)
      qs = list of q values to evaluate scattering intensity at
    '''
    ### initialize sq array for same length of q-range
    nbins = len(qs)
    sq = np.zeros((nbins))
    ### number of atoms in fibril to be simulated
    natoms = len(pos)
    ### calculate contribution to scattering for each pair of atom
    for i in range(natoms):
        ### find displacements from current points to all subsequent points
        ### prevents double counting of displacements
        all_disp = pos[i,:]-pos[(i+1):,:]
        
        ### calculate the distance from displacement vector ###
        rij = np.sqrt(np.sum(np.square(all_disp),axis=1))

        #create array of q vectors and radiuses
        qi = qs[np.newaxis,:]
        R = rij[:,np.newaxis]
        ### add scattering intensity contribution for given pair of atoms
        sq = sq+np.sum(ne.evaluate("2*sin(R*qi)/(R*qi)"),axis=0)

    return sq

def rot_matrix(u,theta):
    '''
    Generates a rotation matrix given a unit vector and angle
    see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    Input
      u = unit vector in 3d cartesian coords about which the rotation will occur
      theta = angle in rad to rotate
    '''
    ux = u[0]
    uy = u[1]
    uz = u[2]
    R = np.zeros((3,3))
    R[0,0] = np.cos(theta)+ux**2*(1-np.cos(theta))
    R[0,1] = ux*uy*(1-np.cos(theta))-uz*np.sin(theta)
    R[0,2] = ux*uz*(1-np.cos(theta))+uy*np.sin(theta)
    R[1,0] = uy*ux*(1-np.cos(theta))+uz*np.sin(theta)
    R[1,1] = np.cos(theta)+uy**2*(1-np.cos(theta))
    R[1,2] = uy*uz*(1-np.cos(theta))-ux*np.sin(theta)
    R[2,0] = uz*ux*(1-np.cos(theta))-uy*np.sin(theta)
    R[2,1] = uz*uy*(1-np.cos(theta))+ux*np.sin(theta)
    R[2,2] = np.cos(theta)+uz**2*(1-np.cos(theta))
    
    return R


def gen_snake_avoid(sigma, length, unitL, diam, verbose=False):
    '''
    Generates a self avoiding "snake" as a set of backbone
    points in 3d cartesian coords. Self-avoidance is determined
    using the defined diameter

    Input
      sigma = std. deviation the angle (about 0) chosen for each unit length bend (rad)
      length = total contour length of the snake (Å)
      unitL = unit length (Å)
      diam_outer = diameter of desired semiflexible cylinder
      verbose = whether or not you want to have printed output of rejected bends
    '''
    restarts = 0
    # setup arrays and countour length variable
    cont_length = int(length/unitL)
    coords = []
    coords_good = []
    css = []
    axs = []
    # iterate over each unit in the contour length
    # this makes a rigid cylinder "snake" in z direction
    for i in range(cont_length):
        coords.append([0,0,i])
        css.append([[1,0,0],[0,1,0]])
        axs.append([0,0,1])
    # make sure to set arrays as floats
    coords = np.asarray(coords, dtype = float)
    css = np.asarray(css, dtype = float)
    axs = np.asarray(axs, dtype = float)
    # set up a counter
    count_index = 0
    # starting point at origin
    current = np.array([0,0,0], dtype = float)
    # this array will be accepted points in the snake backbone
    coords_good = []
    # this loop goes through each point of our rigid snake and bends it
    # set up count fails variable
    i = 0
    while i < cont_length:
        # increase index counter by 1
        count_index += 1
        # remakes the coords array by concatenating unbent section with bent section
        if i > 1:
            coords = np.vstack((coords_old, coords_proj))
        # appends accepted point into our output array
        coords_good.append(current)
        # grabs cross section and axial unit vectors for current points
        cs_unit = css[i]
        ax_unit = axs[i]
        # self avoidance check variable
        self_avoid = False
        # this loop bends the snake and checks for self intersections
        count_fails = 0
        while self_avoid == False:
            count_fails += 1
            # generate random angles for unit vector in spherical coordinates
            # see https://mathworld.wolfram.com/SpherePointPicking.html for explanation
            v_theta = np.arccos(2*np.random.random()-1) # polar angle
            v_phi = 2*np.pi*np.random.random() # azimuthal angle
            # convert from spherical to cartesian
            # rot_u is the unit vector describing rotation axis
            rot_u = np.array([np.cos(v_phi)*np.sin(v_theta),
                              np.sin(v_phi)*np.sin(v_theta),
                              np.cos(v_theta)])
            # rot_theta is the angle (max of pi) which we will rotate around axis rot_u
            rot_theta = np.absolute(np.random.normal(0,sigma))
            if rot_theta > np.pi:
                rot_theta = np.pi
            # generate the rotation matrix
            rot_mat = rot_matrix(rot_u,rot_theta)
            # rotate the axial and cross section unit vectors for current backbone point
            ax_unit = np.matmul(rot_mat,ax_unit)
            cs_unit = np.array([np.matmul(rot_mat,c) for c in cs_unit])
            # assign all following backbone points the same unit vectors
            axs[i:,:] = ax_unit
            css[i:,:,:] = cs_unit
            # projected coordinates of following backbone points (including current point)
            # this projection is just a rigid rod in direction of the current point
            coords_proj = []
            coords_proj.append(current)
            length_remain = cont_length - i
            for j in range(1,length_remain):
                coords_proj.append(current+ax_unit*unitL*j)
            coords_proj = np.asarray(coords_proj)
            # coordinates of untouched preceding backbone points (excluding current point)
            coords_old = []
            for k in range(i):
                coords_old.append(coords[k])
            coords_old = np.asarray(coords_old)
            # find any intersections between projected pts and preceding pts
            # indexes is a counter for number of intersections 
            indexes = 0
            if i > 1:
                # KD tree used for faster searching
                # removing diam*1.4 number of points around the current point to avoid
                # false "intersections" of neigboring points
                kd_tree_proj = KDTree(coords_proj[int(diam*0.7):,:])
                kd_tree_old = KDTree(coords_old[:int(diam*(-0.7)),:])
                indexes = kd_tree_proj.count_neighbors(kd_tree_old, r=diam, cumulative=False)
            # if no intersections are found avoidance is true and on to next index
            if indexes == 0:
                self_avoid = True
                # new current point based on previous point's axial unit vector
                current = current+ax_unit*unitL
            else:
                self_avoid = False # there are intersections, repeat loop again
            if count_fails == 2000:
                restarts +=1
                # print('restart number ' + str(restarts))
                # Restart from the beginning
                coords = []
                coords_good = []
                css = []
                axs = []
                # iterate over each unit in the contour length
                # this makes a rigid cylinder "snake" in z direction
                for i in range(cont_length):
                    coords.append([0,0,i])
                    css.append([[1,0,0],[0,1,0]])
                    axs.append([0,0,1])
                # make sure to set arrays as floats
                coords = np.asarray(coords, dtype = float)
                css = np.asarray(css, dtype = float)
                axs = np.asarray(axs, dtype = float)
                # set up a counter
                count_index = 0
                # starting point at origin
                current = np.array([0,0,0], dtype = float)
                # this array will be accepted points in the snake backbone
                coords_good = []
                i = -1
                self_avoid = True
        i += 1
    print('restarted ' + str(restarts) + ' times')
    return coords_good, axs, css

def contour_length(points):
    """
    Compute the contour length of an ordered array of 3D cartesian points.
    
    Parameters:
    - points: a list or numpy array of shape (N, 3), where N is the number of points.
    
    Returns:
    - The contour length of the curve described by the points.
    """
    # Ensure the input is a numpy array
    points = np.array(points)
    # Calculate the differences between consecutive points
    differences = np.diff(points, axis=0)
    # Calculate the Euclidean distance for each difference
    distances = np.linalg.norm(differences, axis=1)
    # Return the sum of the distances
    return np.sum(distances)

def contour_length_array(points):
    """
    Compute the cumulative contour length at each point of an ordered array of 3D cartesian points
    
    Parameters:
    - points: a list or numpy array of shape (N, 3), where N is the number of points.
    
    Returns:
    - an array of cumulative contour length of the curve at each point in "points".
    """
    # Compute the differences between consecutive points
    diffs = np.diff(points, axis=0)
    
    # Compute the Euclidean distance for each difference
    distances = np.linalg.norm(diffs, axis=1)
    
    # Compute the cumulative sum of distances
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    return cumulative_distances


def gen_snake_scat(e_dens,coords,css,diam):
    '''
    Populates a "snake" defined by points along backbone axis
    and diameter with scattering points to output an array of
    cartesian points.

    Input
      nscat = number of scatterers
      coords = coordinates of snake backbone
      css = array of cross section unit vectors shape N,2,3
      diam = diameter of desired semiflexible cylinder
    '''
    # N is L/UnitL
    N = len(coords)
    scat_coords = []
    # nscat is total number of scatterers
    worm_length = contour_length(coords)
    # electron_dens = nscat_core/(worm_length*np.pi*(diam/2)**2)
    nscat = int(e_dens*worm_length*np.pi*(diam/2)**2)
    for i in range(nscat):
        # selects a random integer in range [0,N-1)
        interval = int(np.floor(np.random.random()*(N-1)))
        # selects a random position between neighboring coords
        t = np.random.random()
        axis_pos = coords[interval]*t+coords[interval+1]*(1-t)
        # random radius (sqrt for uniform density of points) and theta
        rand_r = np.sqrt(np.random.random()) * diam/2
        rand_theta = np.random.random() * 2 * np.pi
        # convert to cartesian
        rand_x = rand_r * np.cos(rand_theta)
        rand_y = rand_r * np.sin(rand_theta)
        # find weighted average css unit vectors between the two neighboring points
        css_x = css[interval][0]*t+css[interval+1][0]*(1-t)
        css_x_norm = css_x/np.linalg.norm(css_x)
        css_y = css[interval][1]*t+css[interval+1][1]*(1-t)
        css_y_norm = css_y/np.linalg.norm(css_y)
        # populate scat_coords list with new coordinate
        scat_coords.append(axis_pos + css_x_norm*rand_x + css_y_norm*rand_y)
        
    return np.array(scat_coords)

def gen_scat_coords_flexcyl(e_dens,length,unitL,diam,sigma,save_dir,i,verbose=False):
    '''
    Generates a flexible cylinder using gen_snake_avoid() and gen_snake_scat().
    Returns an array of scattering points as 3D cartesian coordinates of units Å

    Input
      nscat = 
      length = total contour length of the snake (Å)
      unitL = unit length (Å)
      diam = diameter of desired semiflexible cylinder
      sigma = std. deviation the angle (about 0) chosen for each unit length bend (rad)
      fuzz_length = length (radius-like) of fuzziness. ex: 10Å side chains extending from polymer
      verbose = whether or not you want to have printed output of failed (intersecting) bends
    '''
    coords,axs,css = gen_snake_avoid(sigma,length,unitL,diam,verbose)
    
    save_name_coords = 'flex_backbone_coords' + str(i)
    save_name_axs = 'flex_backbone_axs' + str(i)
    save_name_css = 'flex_backbone_css' + str(i)
    
    np.save(save_dir + save_name_coords + '.npy', coords)
    np.save(save_dir + save_name_axs + '.npy', axs)
    np.save(save_dir + save_name_css + '.npy', css)
    
    
    core_scat = gen_snake_scat(e_dens,coords,css,diam)
    return core_scat

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import imageio
import glob

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def plot_coords(coordinates):
    '''Plots points in 3d cartesian coordinate system

    Input
      coordinates: an N,3 array with x,y,z columns.
    '''
    x = coordinates[:,0]
    y = coordinates[:,1]
    z = coordinates[:,2]

    # x = test[:,0]
    # y = test[:,1]
    # z = test[:,2]

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    # ax.grid(False)
    # ax.set_aspect('equal')

    ax.scatter3D(x, y, z, c=(x**2+y**2+z**2)**0.5, cmap = 'viridis')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    ax.quiver(0, 0, 0, 100, 0, 0, color='tab:blue', arrow_length_ratio=0.05) # x-axis
    ax.quiver(0, 0, 0, 0, 100, 0, color='tab:green', arrow_length_ratio=0.05) # y-axis
    ax.quiver(0, 0, 0, 0, 0, 100, color='tab:red', arrow_length_ratio=0.05) # z-axis

    set_axes_equal(ax)
    # for angle in range(-150,-90):
    #     ax.view_init(35,angle,0)
    #     plt.savefig('/Users/Thomas2/crease_ga/worm_chain_movie/chain2_' + str(angle) + '.png', dpi=200)
    # ax.view_init(35, -90, 0)

    plt.show()
    
def plot_coords_movie(coordinates, path):
    '''Plots points in 3d cartesian coordinate system

    Input
      coordinates: an N,3 array with x,y,z columns.
    '''
    x = coordinates[:,0]
    y = coordinates[:,1]
    z = coordinates[:,2]

    # x = test[:,0]
    # y = test[:,1]
    # z = test[:,2]

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    # ax.grid(False)
    # ax.set_aspect('equal')

    ax.scatter3D(x, y, z, c=(x**2+y**2+z**2)**0.5, cmap = 'viridis')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    ax.quiver(0, 0, 0, 100, 0, 0, color='tab:blue', arrow_length_ratio=0.05) # x-axis
    ax.quiver(0, 0, 0, 0, 100, 0, color='tab:green', arrow_length_ratio=0.05) # y-axis
    ax.quiver(0, 0, 0, 0, 0, 100, color='tab:red', arrow_length_ratio=0.05) # z-axis

    set_axes_equal(ax)
    for angle in range(-180,-90,5):
        ax.view_init(35,angle,0)
        plt.savefig(path + 'worm_' + str(angle) + '.png', dpi=200)
    ax.view_init(35, -90, 0)
    
def generate_and_save_scat_coords(e_dens, length, unit_length, diam, sigma, save_dir, i):
    '''
    Generates scattering coordinates array for a specified fibril. 
    Wrapper for gen-scat_coords_flexcyl() 

    Inputs
      num_scatterers - number of scatterers (calculated based off electron density and volume)
      length - contour length Å
      unit_length - backbone unit length (usually 1Å)
      diameter - fibril diameter Å
      sigma - flexibility value given as gaussian std deviation of angle about 0 in radians
      save_dir - string save directory
    '''
    coords = gen_scat_coords_flexcyl(e_dens,length,unit_length,diam,sigma,save_dir,i,verbose=False)
    save_name = 'flex_cylinder_num' + str(i)
    np.save(save_dir + save_name + '.npy', coords)
            
def test_fibril_scattering_par(num_fibrils,length,diam,sigma,e_dens,exp_path,save_dir):
    '''
    Generates scattering coordinates arrays and the scattering I vs Q 
    for multiple fibrils of the same characteristics.
    Utilizes parallelization.

    Inputs
      num_fibrils - number of fibrils to simulate
      length - contour length Å
      diameter - fibril diameter Å
      sigma - flexibility value given as gaussian std deviation of angle about 0 in radians
      electron_dens - electron density given as (e-/Å3)
      q_norm_idx - index where you would like simulated I vs Q normalized to experimental data
      exp_path - experimental datapath 
      save_dir - string save directory
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    unit_length = 1

    # Specify the file path where you want to save the parameters
    file_path = save_dir + 'parameters.txt'

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each parameter to a new line in the file
        file.write(f"length: {length}\n")
        file.write(f"unit_length: {unit_length}\n")
        file.write(f"diameter: {diam}\n")
        file.write(f"sigma: {sigma}\n")
        file.write(f"electron_dens: {e_dens}\n")

    # calculate number of scatterers based off fibril characteristics
    # num_scatterers = (electron_dens*length*np.pi*(diameter/2)**2)
    # num_scatterers = int(num_scatterers)
    for i in range(num_fibrils):
        generate_and_save_scat_coords(e_dens, length, unit_length, diam, sigma, save_dir, i)
        
    worm_dir = save_dir + 'worm_movie/'
    coords = np.load(save_dir + 'flex_cylinder_num0.npy')
    if not os.path.exists(worm_dir):
        os.makedirs(worm_dir)
    plot_coords_movie(coords, worm_dir)
    images = []
    for filename in sorted(glob.glob(worm_dir + '*.png')):
        images.append(imageio.imread(filename))
    imageio.mimsave(worm_dir + 'chain_movie.gif', images)

    exp_data = np.loadtxt(exp_path)
    qs = exp_data[:,0]

    for i in range(num_fibrils):
        coords = np.load(save_dir + 'flex_cylinder_num' + str(i) + '.npy')
        ints = sq(coords, qs)
        sim_data = np.column_stack((qs, ints))
        np.savetxt(save_dir + 'flex_cylinder_num' + str(i) + '_QI.txt', sim_data)

    fig, ax1 = subplots(1)
    markers, caps, bars = ax1.errorbar(exp_data[:,0], exp_data[:,1], yerr=exp_data[:,2],
     color = 'black', label ='Exp Data', linestyle='None', marker = '.', markersize=3)
    [bar.set_alpha(0.05) for bar in bars]
    [cap.set_alpha(0.05) for cap in caps]

    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.cool(np.linspace(0, 1, num_fibrils))))
    for i in range(num_fibrils):
        sim_data = np.loadtxt(save_dir + 'flex_cylinder_num' + str(i) + '_QI.txt')
        ax1.plot(sim_data[:,0], sim_data[:,1], label ='Sim Data' + str(i), linestyle='None', marker = '.', markersize=3)

    ax1.set_xlabel('q ($\AA^{-1}$)')
    ax1.set_ylabel('Intensity ($cm^{-1}$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)
    ax1.legend(prop={'size': 12})
    plt.tight_layout()
    plt.savefig(save_dir + 'IQ_comparison.png', dpi=300)
    plt.close('all')

savepath = '/projects/thch7683/modeling_test/'
exp_path = '/projects/thch7683/test_files/PM7_CN_sol618_buf617_emp614_log_thin200.txt'

num_fibrils = 1
lengths = [2000, 1500, 1000, 500]
# diams = np.arange(25,1050,25)
diam = 25
flexes = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]
e_dense = 0.0001

flexes_dict = {0.2:'0p2', 0.1:'0p1', 0.05:'0p05', 0.01:'0p01',0.005:'0p005', 
0.001:'0p001', 0.0001:'0p0001'}
dens_dict = {0.0001:'0p0001', 0.01:'0p01'}
for length in lengths:
    for flex in flexes:
        flex_label = flexes_dict[flex]
        dens_label = dens_dict[e_dense]
        test_fibril_scattering_par(
            num_fibrils,length,diam,flex,e_dense,exp_path,
            savepath + 'nofuzz_fibril_oct23/' + f'dtotal{diam}_length{length}_flex{flex_label}_edens{dens_label}/'
            )