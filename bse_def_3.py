import numpy as np
import os
import pickle
import h5py
import math
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from itertools import permutations

#%% Function definitions

"""Library of definitions needed for Bloch state expansion calculations"""

def _getline(cube):
    """
    Read a line from cube file where first field is an int 
    and the remaining fields are floats.

    params:
        cube: file object of the cube file

    returns: (int, list<float>)
    """
    l = cube.readline().strip().split()
    return int(l[0]), [float(x) for x in l[1:]]

def read_dummy_cube(path):
    meta = {}
    with open(path, 'r') as cube:
        # Skip header lines
        cube.readline()
        # Read grid dimensions and origin
        natm, meta['org'] = _getline(cube)
        # Read x,y,z coordinate information
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)
        meta['N_x'] = nx
        meta['N_y'] = ny
        meta['N_z'] = nz
        # Read data
        data = np.zeros((nx*ny*nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
        data = np.reshape(data, (nx, ny, nz))
    return data, meta


def read_cube(fname):
    """
    Reads all information from the cube file. Returns the metadata on coordinates, the Z-number and coordinates of the atoms,
    and the actual info in the cube file in a 3D array organized along x, y and z coordinates.
    """
    meta = {}
    no_error = True
    try:
        with open(fname, 'r') as cube:
            cube.readline(); cube.readline()  # ignore comments
            #read coordinate data
            natm, meta['org'] = _getline(cube)        # reads number of atoms and coordinates of the origin
            nx, meta['xvec'] = _getline(cube)         # reads number of x steps and length per step
            ny, meta['yvec'] = _getline(cube)         # reads number of y steps and length per step
            nz, meta['zvec'] = _getline(cube)         # reads number of z steps and length per step
            meta['N_x'] = nx
            meta['N_y'] = ny
            meta['N_z'] = nz
            meta['N']=natm
            atom_Z = np.arange(natm)                  # array for storing atomic Z numbers
            atom_pos = np.zeros((natm,3))             # array for storing atomic coordinates - [x,y.z] per atom
            #read information on the atoms
            for i in range(natm):
                line = cube.readline().strip().split()
                atom_Z[i]=int(line[0])
                atom_pos[i,:]=(line[2:])
            #read data from cube file
            data = np.zeros((nx*ny*nz))
            idx = 0
            for line in cube:
                for val in line.strip().split():
                    data[idx] = float(val)
                    idx += 1
        data = np.reshape(data, (nx, ny, nz))
    except FileNotFoundError:
        no_error = False
        data, atom_Z, atom_pos = np.array([]), np.array([]), np.array([])
    return data, meta, atom_Z, atom_pos, no_error

def log_output(log_string, log_file):

    print(log_string)
    with open(log_file, 'a') as log:
        log.write(log_string + '\n')
    return

def set_index(e_k):

    """function returning the crystallographic indices of a direction in reciprocal space characterized by the direction vector e_k """

    abs_e_k = abs(e_k)
    abs_e_k_non_zero = abs_e_k[abs_e_k != 0]
    index = e_k / np.min(abs_e_k_non_zero)
    label = np.sum(abs(index)) - 1
    print(int(label))
    return index, int(label)

def k_path_segment(k1, k2, e_k_dict, dk_set):

    """function that initializes a segment of the k-path. Segment is defined as part of the line vec(k_0) + kappa . vec(e_k)
    where vec(e_k) gives the direction of the line and vec(k_0) is the position vector of the line, which is taken as 
    perpendicular to vec(e_k)"""

    e_k = k2 - k1
    norm_k = np.linalg.norm(e_k)
    e_k /= norm_k                                                              #normalized unit vector along the segment
    index, label = set_index(e_k)                                              #identify the segment direction and get the direction label                

    sign = 1
    vectors = e_k_dict[label]['positive_dir']
    N_d = np.size(vectors, axis=0)
    #print(N_d)
    for v in range(N_d):
        #print(abs(np.dot(e_k, vectors[v, :])))
        if (abs(np.dot(e_k, vectors[v, :])) > 0.99999):
            sign = round(np.dot(e_k, vectors[v, :]))
        #print(sign)

    kappa_1 = np.dot(k1, e_k)                                                  #initial parallel position of the segment in BZ1
    kappa_2 = np.dot(k2, e_k)                                                  #final parallel position of the segment in BZ1
    k_0 = k1 - kappa_1 * e_k                                                   #get position vector of the line                                                  
    N_k = closest((kappa_2 - kappa_1) / dk_set)
    dk_real = (kappa_2 - kappa_1) / N_k
    kappa = np.linspace(kappa_1, kappa_2 - dk_real, N_k)
    if (label == 0):
        angle_1, angle_2 = 0, 0
    else:
        angle_1 = - np.arcsin(sign * e_k[1]/np.sqrt((e_k[0]**2 + e_k[1]**2)))
        angle_2 = - np.arcsin(sign * e_k[2])
    angle = np.array([angle_1, angle_2])
    k_perp_basis = np.array([[np.cos(angle[1])*np.cos(angle[0]), -np.cos(angle[1])*np.sin(angle[0]), -np.sin(angle[1])],\
                             [np.sin(angle[0]), np.cos(angle[0]), 0],\
                             [np.sin(angle[1])*np.cos(angle[0]), -np.sin(angle[1])*np.sin(angle[0]), np.cos(angle[1])]])
    if (label == 0):
        k_perp_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    k_segment={'dir': label, 'scale': e_k_dict[label]['scale'], 'dir_k': e_k, 'indices': index,\
               'pos_k': k_0, 'kappa_12': ([kappa_1, kappa_2]), 'r_par_sign': sign, 'dk': dk_real,\
               'rot_angle': angle, 'k_perp_basis': k_perp_basis, 'N_bun': e_k_dict[label]['N_bun'],\
               'origin_bun': e_k_dict[label]['origin_bun'],'kappa': kappa, 'phi_folded': np.zeros(N_k)}  
    return k_segment

def set_r_para_sign(direction, direction_vector, e_k_dict):

    """ function that sets the sign of the direction_vector, relative to the positive directions as defined
    in e_k_dict
    """

    sign = 1
    for d in range(3):
        vectors = e_k_dict[d]['positive_dir']
        N_d = np.size(vectors, axis=0)
        for v in range(N_d):
            if (abs(np.dot(direction_vector, vectors[v, :])) == 1):
                sign = int(np.dot(direction_vector, vectors[v, :])) 
    return sign

def summary_k_path(k_path, segments, a_dx):

    """This function summarizes the path through BZ1 by identifying the number of (100), (110) and (111) segments.
    A dictionary is returned that contains the total of each of these segments, and the position of these different segments
    in the path. """

    n_dir = np.array([0, 0, 0], dtype=int)                                     #lists number of segments along (100), (110) and (111)
    pos_dir = np.zeros((3, segments))                                          #lists positions of different (100), (110) and (111) segments in path                                                                         
    for s in range(segments):
        segment = k_path[s]
        d = segment['dir']                                                     #identify segment as 0 = (100), 1 = (110), 2 - (111)
        pos_dir[d, n_dir[d]] = s                                               #store position of the segment
        n_dir[d] += 1                                                          #add 1 to total number of relevant segments
    Nyq = math.floor(0.5 * ((a_dx/(np.sqrt(3))) - 2)) 
    #Nyq = 3
    k_structure = {'n_dir': n_dir, 'pos_dir': pos_dir, 'Nyquist': Nyq}

    return k_structure

def rotate_psi(psi, k_segment):

    """function that orients the wavefunction in line with the direction. The reorientation creates a cubic grid aligned with the
    segment axis. For (100), re-orientation is avoided through a re-assignment of the axes. The function fills in the parallel and perpendicular coordinates in line with the wavefunctions. Orientation of the
    corresponding wavenumbers is done upon definition of the bundle."""

    # set orientation and real space coordinates for (100) orientation as initial values for the quantities the function will return
    psi_rot = psi
    N = np.size(psi, axis=0)
    r_par, r_perp_0, r_perp_1 = np.linspace(0, N-1, N), np.linspace(0, N-1, N), np.linspace(0, N-1, N)
    axis_id = np.array([0, 1, 2], dtype=int)
    dir_k = k_segment['dir_k']
    direction = k_segment['dir']
    # in case of (100), provide correct axis assignment for (010) or (001) segments
    if (direction == 0):
        if (abs(dir_k[1]) > 0.999999):
            axis_id = np.array([1, 0, 2], dtype=int)
        elif (abs(dir_k[2]) > 0.999999):
           axis_id = np.array([2, 1, 0], dtype=int)
    # rotations in the case of (110)
    if (direction == 1):
        angle_1 = np.degrees(k_segment['rot_angle'][0])
        angle_2 = np.degrees(k_segment['rot_angle'][1])
        if (abs(dir_k[2]) < 1e-6):
            psi_rot = rotate(psi, angle=angle_1, axes=(0,1), reshape=True)         #rotate 45 degrees around the y-axis
        elif (abs(dir_k[1]) < 1e-6):
            psi_rot = rotate(psi, angle=angle_2, axes=(0,2), reshape=True)
        elif (abs(dir_k[0]) < 1e-6):
            psi_i = rotate(psi, angle=angle_1, axes=(0,1), reshape=True)              #first rotation for (111), -45 degrees around z axis
            psi_rot = rotate(psi_i, angle=angle_2, axes=(0,2), reshape=True)          #second rotation for (111), -35.2 degrees around y axis
        N_par = np.size(psi_rot, axis=0)
        r_par = np.linspace(0, N_par-1, N_par)
        N_perp_0 = np.size(psi_rot, axis=1)
        r_perp_0 = np.linspace(0, N_perp_0-1, N_perp_0)
        N_perp_1 = np.size(psi_rot, axis=2)
        r_perp_1 = np.linspace(0, N_perp_1-1, N_perp_1)
    # set sign of parallel coordinate according to sign of direction
    r_par *= k_segment['r_par_sign']
    # other orientations still to program

    return psi_rot, r_par, r_perp_0, r_perp_1, axis_id

def rotate_psi_111(psi, k_segment):

    """function that orients the wavefunction in line with the given 110 direction. The reorientation creates a cubic grid aligned with the
    segment axis. The function fills in the parallel and perpendicular coordinates in line with the wavefunctions. Orientation of the
    corresponding wavenumbers is done upon definition of the bundle. For (111), angles are -45 and -35.2"""

    angle = k_segment['rot_angle']
    angle_1, angle_2 = np.degrees(angle[0]), np.degrees(angle[1])
    sign = k_segment['r_par_sign']
    psi_i = rotate(psi, angle=sign * angle_1, axes=(0,1), reshape=True)              #first rotation for (111), -45 degrees around z axis
    psi_rot = rotate(psi_i, angle=sign * angle_2, axes=(0,2), reshape=True)          #second rotation for (111), -35.2 degrees around y axis
    axis_id = np.array([0, 1, 2], dtype=int)
    N_par = np.size(psi_rot, axis=axis_id[0])
    r_par = np.linspace(0, N_par-1, N_par)
    N_perp_0, N_perp_1 = np.size(psi_rot, axis=axis_id[1]), np.size(psi_rot, axis=axis_id[2])
    r_perp_0, r_perp_1 = np.linspace(0, N_perp_0-1, N_perp_0), np.linspace(0, N_perp_1-1, N_perp_1) 

    return psi_rot, sign * r_par, r_perp_0, r_perp_1, axis_id

def set_bundle_100(b, k_segment, k_structure):

    #identify direction and set in-plane coordinates
    dir_index = k_segment['indices']
    #direction = k_segment['dir']
    mask_1, mask_2 = get_masks(k_segment['dir'], dir_index)
    N = k_structure['Nyquist']
    if (b == 0):
        k_0_0, k_0_1 = np.linspace(-N, N, 2 * N + 1), np.linspace(-N, N, 2 * N + 1) 
    else:
        k_0_0, k_0_1 = np.linspace(-N, N - 1, 2 * N), np.linspace(-N, N - 1, 2 * N)
        #print(k_0_0)
    k_0 = k_segment['pos_k']
    k_0_perp = np.array([np.dot(mask_1, k_0), np.dot(mask_2, k_0)])
    #create array with k0 coordinates of the bundle
    k0_origin = k_segment['origin_bun'][b]
    k0_origin_perp = np.array([np.dot(mask_1, k0_origin), np.dot(mask_2, k0_origin)])
    k_0_0 += k0_origin_perp[0] + k_0_perp[0]  
    k_0_1 += k0_origin_perp[1] + k_0_perp[1]  
    #set kappa array NOTE shift dk, N_k, kappa_BZ0 to segment definition!
    N_k = np.size(k_segment['kappa'])
    dk = k_segment['dk']
    #kappa_0 = k_segment['kappa_0_bun'][b]                                                # Works OK for square lattice of BZs
    if (b == 0):
        kappa = np.zeros((N_k, 2 * N + 1))
        for i in range(2 * N + 1):
            N_zone = - N + i 
            kappa_1 = k_segment['kappa_12'][0] + N_zone
            kappa_2 = k_segment['kappa_12'][1] + N_zone
            kappa[:, i] = np.linspace(kappa_1, kappa_2 - dk, N_k) 
            #print(kappa_1), print(kappa_2), print(dk)
            #print(kappa[:, i])
    else:
        kappa = np.zeros((N_k, 2 * N))
        for i in range(2 * N):
            N_zone = - N + 1/2 + i 
            kappa_1 = k_segment['kappa_12'][0] + N_zone
            kappa_2 = k_segment['kappa_12'][1] + N_zone
            kappa[:, i] = np.linspace(kappa_1, kappa_2 - dk, N_k) 
    bundle = {'dir_index': dir_index, 'k_0_0': k_0_0, 'k_0_1': k_0_1, 'kappa': kappa, 'psi_proj': np.array([])}
    return bundle

def combinations_110_bis(k_segment, l, b, Nyq):

    sign = k_segment['r_par_sign']
    index = sign * k_segment['indices']
    n_2 = np.linspace(-Nyq, Nyq - b, 2 * Nyq + 1 - b)
    if (l == 0):
        BZ_2D = np.array([[0, 0]])
    else:
        BZ_2D = np.array([[l, 0], [0, l]])
    pos_y, pos_z = 1, 2
    if (abs(index[1]) < 1e-6):
        pos_y, pos_z = 2, 1
    elif (abs(index[0]) < 1e-6):
        pos_y, pos_z = 2, 0
    repeat_BZ_2D = np.repeat(BZ_2D, len(n_2), axis=0)
    tiled_n_2 = np.tile(n_2, len(BZ_2D)).reshape(-1, 1)
    BZ = np.hstack((repeat_BZ_2D[:, :pos_z], tiled_n_2, repeat_BZ_2D[:, pos_z:]))
    BZ[:, pos_y] *= index[pos_y]
    BZ += k_segment['origin_bun'][b]
    return BZ

def set_bundle_110(BZ, l, b, k_segment, k_structure):

    #identify direction and set in-plane coordinates
    dir_index = k_segment['indices']
    mask_1, mask_2 = get_masks(k_segment['dir'], dir_index)
    N = k_structure['Nyquist']
    basis = k_segment['k_perp_basis']
    #create array with k0 coordinates of the layer in the bundle
    e_par = basis[0]
    e_perp_0, e_perp_1 = basis[1], basis[2] 
    #if (abs(index[2]) < 1e-6):
    #    e_perp_0, e_perp_1 = basis[1], basis[2]
    k_perp_0 = np.sum(BZ * e_perp_0, axis=1)
    k_perp_1 = np.sum(BZ * e_perp_1, axis=1)
    k_0 = k_segment['pos_k']
    k_perp_0 += np.dot(e_perp_0, k_0)
    k_perp_1 += np.dot(e_perp_1, k_0)
    #create array with kappa coordinates of the layer in the bundle
    #Delta_k = np.dot(BZ[0] * e_par)
    #print(BZ[0]), print(e_par), print(Delta_k)
    #set kappa array NOTE shift dk, N_k, kappa_BZ0 to segment definition!
    N_k = np.size(k_segment['kappa'])
    dk = k_segment['dk']
    n_min = l - 2 * N
    n_max = 2 * (N - b) - l
    #print(n_min, n_max)
    n_kappa = round((n_max - n_min) / 2 + 1) 
    kappa = np.zeros((N_k, n_kappa))
    kappa_0 = np.dot(e_par, k_segment['origin_bun'][b])
    #print(kappa_0)
    for i in range(n_kappa):
        N_zone = n_min / 2 + i 
        kappa_1 = kappa_0 + k_segment['kappa_12'][0] + N_zone / k_segment['scale']
        kappa_2 = kappa_0 + k_segment['kappa_12'][1] + N_zone / k_segment['scale']
        kappa[:, i] = np.linspace(kappa_1, kappa_2 - dk, N_k) 
        #print(kappa_1), print(kappa_2), print(dk)
        #print(kappa[:, i])
    bundle = {'dir_index': dir_index, 'k_0_0': k_perp_0, 'k_0_1': k_perp_1, 'kappa': kappa, 'psi_proj': np.array([])}
    return bundle

def combinations_111(width, length, bundle, k_segment):

    sign = k_segment['r_par_sign']
    index = sign * k_segment['indices']
    cell = [width, length, 0]
    all_cells = set(permutations(cell))
    all_cells_array = np.array(list(all_cells))
    if (np.sum(index) < 1.00001):
        if (index[0] > 0.99999) and (index[1] > 0.99999):
            all_cells_array[:,2] *= -1
        elif (index[0] > 0.99999) and (index[2] > 0.99999):
            all_cells_array[:,1] *= -1
        elif (np.sum(index) < -0.99999): 
            all_cells_array[:,1] *= -1
            all_cells_array[:,2] *= -1
    BZ = all_cells_array.astype(float)
    BZ += k_segment['origin_bun'][bundle]
    return BZ

def set_bundle_111(BZ, w, l, b, k_segment, k_structure):

    #identify direction and set in-plane coordinates
    dir_index = k_segment['indices']
    mask_1, mask_2 = get_masks(k_segment['dir'], dir_index)
    N = k_structure['Nyquist']
    basis = k_segment['k_perp_basis']
    #create array with k0 coordinates of the layer in the bundle
    e_par = basis[0]
    e_perp_0, e_perp_1 = basis[1], basis[2] 
    k_perp_0 = np.sum(BZ * e_perp_0, axis=1)
    k_perp_1 = np.sum(BZ * e_perp_1, axis=1)
    k_0 = k_segment['pos_k']
    k_perp_0 += np.dot(e_perp_0, k_0)
    k_perp_1 += np.dot(e_perp_1, k_0)
    #create array with kappa coordinates of the layer in the bundle
    #Delta_k = np.sum(BZ[0] * e_par)
    #print(BZ[0]), print(e_par), print(Delta_k)
    #set kappa array NOTE shift dk, N_k, kappa_BZ0 to segment definition!
    N_k = np.size(k_segment['kappa'])
    dk = k_segment['dk']
    n_min = w + l - 3 * N
    n_max = 3 * (N - b) + l - 2 * w
    #print(n_min, n_max)
    n_kappa = round((n_max - n_min) / 3 + 1) 
    kappa = np.zeros((N_k, n_kappa))
    kappa_0 = np.dot(e_par, k_segment['origin_bun'][b])
    #print(kappa_0)
    for i in range(0, n_kappa):
        n_cell = n_min + 3 * i
        kappa_1 = kappa_0 + k_segment['kappa_12'][0] + n_cell * k_segment['scale']
        kappa_2 = kappa_0 + k_segment['kappa_12'][1] + n_cell * k_segment['scale']
        kappa[:, i] = np.linspace(kappa_1, kappa_2 - dk, N_k) 
        #print(kappa_1), print(kappa_2), print(dk)
        #print(kappa[:, i])
    bundle = {'dir_index': dir_index, 'k_0_0': k_perp_0, 'k_0_1': k_perp_1, 'kappa': kappa, 'psi_proj': np.array([])}
    return bundle

def get_masks(direction, dir_index):

    """Function that provides the conversion of the coordinates of the position vector k_0 before rotation of the wavefunction
       into the (in plane) coordinates after rotation. Note that for (100) and (110), axes are swapped to mimick a rotation""" 

    mask_1, mask_2 = np.array([0, 1, 0]), np.array([0, 0, 1])                  #masks for parallel axis as the x-axis, situation for (100) and (-100) segment.
    #axis_id = np.array([0, 1, 2], dtype=int)
    if (direction == 0):
        if (abs(dir_index[1]) == 1):
            mask_1, mask_2 = np.array([1, 0, 0]), np.array([0, 0, 1])          #masks and axis number for (010) and (0-10) segment            
        elif (abs(dir_index[2]) == 1):
           mask_1, mask_2 = np.array([0, 1, 0]), np.array([1, 0, 0])           #masks and axis number for (001) and (00-1) segment
    if (direction == 1):
        mask_1 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])                    # set masks for (110) direction - selects k_y component of k_pos in rotated coordinate system
        mask_2 = np.array([0, 0, 1])
        if (abs(dir_index[2]) == 0):                                           #identify set of (110) directons
            if ((dir_index[0] + dir_index[1]) == 0):                           #identify (-110) directions
                mask_1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])             #for (1-10), mask selects kx component of k_pos in rotated coordinate system
        if (abs(dir_index[1]) == 0):                                           #identify (101) directions
            mask_1 = np.array([-1/np.sqrt(2), 0, 1/np.sqrt(2)])                # set masks for (110) direction - selects k_y component of k_pos in rotated coordinate system
            mask_2 = np.array([0, 1, 0])
            if ((dir_index[0] + dir_index[2]) == 0):
                mask_1 = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])             #for (-101), same as (1-10)
        if (abs(dir_index[0]) == 0):                                               #identify (011) directions
            mask_1 = np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])                # set masks for (011) direction - selects k_y component of k_pos in rotated coordinate system
            mask_2 = np.array([1, 0, 0])
            if ((dir_index[1] + dir_index[2]) == 0):
                mask_1 = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])  
    if (direction == 2):                                                       # identify set of (111) directions 
        mask_1 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])                    # set mask for (111)
        mask_2 = np.array([-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)])
    return mask_1, mask_2

def closest (x):

    lower = math.floor(x)
    upper = math.ceil(x)
    diff_lower = abs(x - lower)
    diff_upper = abs(x - upper)
    if diff_lower < diff_upper:
        return lower
    else:
        return upper

def find_closest_index(r_range, r):

    absolute_diff = np.abs(r_range - r)
    closest_index = np.argmin(absolute_diff)
    return closest_index

def organize_output_path(k_path, k_path_points):

    """function that provides the predefined path through reciprocal space by concatenating the different segments. Returns
    an array with parallel wavenumbers and the position of the high-symmetry points that define the start and the end of a
    segment"""

    N_segment = np.size(k_path_points, axis=0) - 1                             #Number of segments
    kappa = np.array([0])                                                      #Array containing kappa points - it is assumed that the path in BZ1 is closed
    kappa_ticks = np.zeros(N_segment + 1)                                          #Positioin of the ticks in k-space
    #k_point_name[0] = k_path_names[0]
    kappa_end = 0
    kappa_ticks[0] = kappa_end
    for s in range(N_segment):
        k_segment = k_path[s]
        #Shift the kappa array of the segment and add it to the kappa array of the path 
        dkappa_s = k_segment['dk']
        kappa_s = k_segment['kappa']
        N_kappa_s = np.size(kappa_s)
        kappa_1 = k_segment['kappa'][0]
        kappa_s -= (kappa_1 - kappa_end)
        kappa = np.concatenate((kappa, kappa_s))
        kappa_end = kappa_s[N_kappa_s - 1] + dkappa_s
        kappa_ticks[s+1] = kappa_end
    #set info in the first and last points of the arrays equal
    kappa = kappa[1:]
    kappa = np.concatenate((kappa, ([kappa_end])))    
    return kappa, kappa_ticks

def organize_output_phi(k_path):

    """function merges all projected phi densities in the different segment along the predefined
       path through reciprocal space. A 1D arrays phi_folded is returned that contains the expansion coefficients
       folded back on BZ1 along the kappa axis."""

    N_segment = len(k_path)                                                    #Number of segments
    phi_folded = np.array([0])
    for s in range(N_segment):
        #Add phi_folded of the segment to the phi_folded array of the path
        k_segment = k_path[s]
        phi_folded_segment = k_segment['phi_folded']
        phi_folded = np.concatenate((phi_folded, phi_folded_segment))
    #set info in the first and last points of the arrays equal
    phi_folded = phi_folded[1:]
    phi_folded = np.concatenate((phi_folded, ([phi_folded[0]])))

    return phi_folded

"""
def clip_array(array, n_min, n_max):
    clips array using a border defined by n_min and n_max -- removing elements along all dimensions
    before position n_min and after position n_max

    # Get the shape of the array
    shape = array.shape
    # Create a mask for each dimension
    mask_axis0 = (np.arange(shape[0]) >= n_min) & (np.arange(shape[0]) <= n_max)
    mask_axis1 = (np.arange(shape[1]) >= n_min) & (np.arange(shape[1]) <= n_max)
    mask_axis2 = (np.arange(shape[2]) >= n_min) & (np.arange(shape[2]) <= n_max)
    # Apply the masks to the array
    clipped_array = array[mask_axis0][:, mask_axis1][:, :, mask_axis2]

    return clipped_array
"""
def clip_cube(psi, atom_pos, frame, dx):

    """Clips psi, leaving a border with width frame around the outermost atoms"""

    atom_pos_array = np.array(atom_pos)
    # Get minimum and maximum atom position along 3 directions
    x_min, x_max = np.min(atom_pos_array[:, 0]), np.max(atom_pos_array[:, 0])
    y_min, y_max = np.min(atom_pos_array[:, 1]), np.max(atom_pos_array[:, 1])
    z_min, z_max = np.min(atom_pos_array[:, 2]), np.max(atom_pos_array[:, 2])
    # Turn positions in space into positions in the 3D array
    n_min = np.array([x_min / dx, y_min / dx, z_min / dx])
    n_max = np.array([x_max / dx, y_max / dx, z_max / dx])
    # Turn frame into a position range
    n_frame = np.array([frame / dx, frame / dx, frame / dx]) 
    # Set clip length at either side of the crystal
    clip_min = int(np.round(np.min(n_min - n_frame)))
    clip_max = int(np.max(n_max + frame)) 
    # Clip the array
    psi_clipped = psi[clip_min : clip_max, clip_min : clip_max, clip_min : clip_max]

    return psi_clipped

def bse_cube(file_specifiers, log_file, k_path, kappa_path, data):

    no_error = True
    files_processed = False
    N_data = np.size(kappa_path)
    bse_folded_states = np.zeros((1, N_data))
    N_cube = file_specifiers['N_cube']
    cube_0 = file_specifiers['cube_0']
    frame = data['frame']
    clip = data['clip']
    state_nr = np.array([0])
    for cube in range(N_cube):
        cube_i = cube_0 + cube
        log_string = 'state ' + str(cube_i) + ' started'
        log_output(log_string, log_file)
        file = file_specifiers['Project'] + '_' + file_specifiers['State'] + file_specifiers['WFN'] +\
               str(cube_i) + file_specifiers['Addition'] + '.' + file_specifiers['extension'][0]
        log_string = 'try open ' + file
        log_output(log_string, log_file)
        psi, meta, atom_Z, atom_pos, no_error = read_cube(file)
        if no_error:
            log_string = 'cube file loaded'
            log_output(log_string, log_file)
            data['dx'] = meta['xvec'][0]
            if clip:
                psi = clip_cube(psi, atom_pos, frame, data['dx'])
            phi_folded_path = bse(psi, k_path, data, log_file)
            bse_folded_states = np.vstack((bse_folded_states, phi_folded_path))
            state_nr = np.concatenate((state_nr, [cube_i]))
            log_string = 'state ' + str(cube_i) + ' analyzed'
            log_output(log_string, log_file)
            files_processed = True
        else:
            log_string = 'ERROR - state ' + str(cube_i) + ', analysis skipped'
            log_output(log_string, log_file)
            continue
    if files_processed:
        bse_folded_states = bse_folded_states[1:]
        state_nr = state_nr[1:]
    return bse_folded_states, state_nr, files_processed

def bse_h5(file_specifiers, log_file, k_path, kappa_path, data):

    no_error = True
    files_processed = False
    N_data = np.size(kappa_path)
    bse_folded_states = np.zeros((1, N_data))
    clip = data['clip']
    frame = data['frame']
    state_nr = np.array([0])
    state_0 = file_specifiers['cube_0']
    h5_file = file_specifiers['h5_file'] + '.' + file_specifiers['extension'][1]
    log_output('search for ' + h5_file, log_file)
    try:
        with h5py.File(h5_file, 'r') as h5:
            psi_all = h5['psi_r'][:]
            atom_pos = h5['atoms'][:]
            dr = h5['grid_spacing'][:]
            print(dr)
    except FileNotFoundError:
        no_error = False
        log_output(h5_file + ' not found', log_file)
    if no_error:
        log_string = 'h5 file loaded'
        log_output(log_string, log_file)
        N_states = np.size(psi_all, axis=0)
        #N_states = 5
        data['dx'] = dr[0]
        for state in range(N_states):
            psi = psi_all[state]
            if clip:
                psi = clip_cube(psi, atom_pos, frame, data['dx'])
            phi_folded_path = bse(psi, k_path, data, log_file)
            bse_folded_states = np.vstack((bse_folded_states, phi_folded_path))
            state_nr = np.concatenate((state_nr, [state + state_0]))
            log_string = 'state ' + str(state) + ' analyzed'
            log_output(log_string, log_file)
            files_processed = True
    else:
        log_string = 'ERROR - BSE analysis stopped'
        log_output(log_string, log_file)
    if files_processed:
        bse_folded_states = bse_folded_states[1:]
        state_nr = state_nr[1:]
    return bse_folded_states, state_nr, files_processed

def bse(psi, k_path, data, log_file):

    # Get calculation data
    a = data['latt_par']
    dx = data['dx']
    k_unit = data['k_unit']
    # Prepare for BSE along pre-defined path    
    segments = len(k_path)
    k_structure = summary_k_path(k_path, segments, a / dx)
    Nyq = k_structure['Nyquist']    
    n_100 = k_structure['n_dir'][0]
    n_110 = k_structure['n_dir'][1]
    n_111 = k_structure['n_dir'][2]
    # expansion for the 100 segments
    for segment in range(n_100):
        #Determine the projected wavefunction, with folding perpendicular to e_k
        pos_100 = int(k_structure['pos_dir'][0, segment])
        k_segment = k_path[pos_100]
        k_segment['phi_folded'][:] = 0
        bundles = k_segment['N_bun']
        psi_rot, r_par, r_perp_0, r_perp_1, axis_id = rotate_psi(psi, k_segment)
        for b in range(bundles):
            # initiate the bundle
            bundle = set_bundle_100(b, k_segment, k_structure)
            # project wavefunction on each line in the bundle
            k0_perp_0, k0_perp_1 = bundle['k_0_0'], bundle['k_0_1']  
            k00_x_rp0 = np.outer(k0_perp_0, r_perp_0) * k_unit * dx
            k01_x_rp1 = np.outer(k0_perp_1, r_perp_1) * k_unit * dx
            factor_perp = np.exp(-1j * (k00_x_rp0[:, :, None, None] + k01_x_rp1[None, None, :, :]))
            # Project and store in a 3D array giving for k0_0 and k0_1 combination the projected wavefunction
            psi_projection_unfolded = np.tensordot(factor_perp, psi_rot, axes=([1, 3], [axis_id[1], axis_id[2]]))   
            # determine Fourier transform along each line in the bundle
            kappa = bundle['kappa']
            factor_par = np.exp(-1j*(k_unit * kappa[:, :, None] * r_par[None, None, :]) * dx) 
            phi_projection_unfolded = np.tensordot(factor_par, psi_projection_unfolded, axes=([2], [2]))
            k_segment['phi_folded'] += np.sum(abs(phi_projection_unfolded)**2, axis=(1, 2, 3))  
    log_string = str(n_100) + ' 100 segments processed'
    log_output(log_string, log_file)
    # Expansion for the 110 segments
    for segment in range(n_110):
        #Determine the projected wavefunction, with folding perpendicular to e_k
        pos_110 = int(k_structure['pos_dir'][1, segment])
        k_segment = k_path[pos_110]
        k_segment['phi_folded'][:] = 0
        bundles = k_segment['N_bun']
        psi_rot, r_par, r_perp_0, r_perp_1, axis_id = rotate_psi(psi, k_segment) 
        #Analyze the different sets of lines (layers) in the bundle, where each layer has a specific repetition of cells
        #in the parallel direction
        for b in range(bundles):
            layers = 2 * Nyq + 1 - b
            for l in range(layers):
                BZ = combinations_110_bis(k_segment, l, b, Nyq)
                # initiate the bundle
                bundle = set_bundle_110(BZ, l, b, k_segment, k_structure)
                # project wavefunction on each line in the bundle
                k0_perp_0, k0_perp_1 = bundle['k_0_0'], bundle['k_0_1']  
                #if (np.size(k0_perp_0) == 0):
                #    continue
                factor_perp = np.exp(-1j * ((k0_perp_0[:, None, None] * r_perp_0[None, :, None]\
                                            + k0_perp_1[:, None, None] * r_perp_1[None, None, :]) * k_unit * dx))
                # Project and store in a 2D array giving for each k0_0, k0_1 combination the projected wavefunction
                psi_projection_unfolded = np.tensordot(factor_perp, psi_rot, axes=([1, 2], [axis_id[1], axis_id[2]]))   
                # determine Fourier transform along each line in the bundle
                # phase factor array for the separate sections on a line in the bundle
                kappa = bundle['kappa']
                factor_par = np.exp(-1j*(k_unit * kappa[:, :, None] * r_par[None, None, :]) * dx) 
                phi_projection_unfolded = np.tensordot(factor_par, psi_projection_unfolded, axes=([2], [1]))
                k_segment['phi_folded'] += np.sum(abs(phi_projection_unfolded)**2, axis=(1, 2))
    log_string = str(n_110) + ' 110 segments processed'
    log_output(log_string, log_file)
    for segment in range(n_111):
        #Determine the projected wavefunction, with folding perpendicular to e_k
        pos_111 = int(k_structure['pos_dir'][2, segment])
        k_segment = k_path[pos_111]
        k_segment['phi_folded'][:] = 0
        bundles = k_segment['N_bun']
        psi_rot, r_par, r_perp_0, r_perp_1, axis_id = rotate_psi_111(psi, k_segment)  
        for b in range(bundles):
            width = 2 * Nyq + 1 - b
            #for w in range(0, 1, 1):
            for w in range(width):
                #for l in range(0, 1, 1):
                for l in range(w + 1):
                    BZ = combinations_111(w, l, b, k_segment)
                    #print(BZ)
                    bundle = set_bundle_111(BZ, w, l, b, k_segment, k_structure)
                    # project wavefunction on each line in the bundle
                    k0_perp_0, k0_perp_1 = bundle['k_0_0'], bundle['k_0_1']  
                    #if (np.size(k0_perp_0) == 0):
                    #    continue
                    factor_perp = np.exp(-1j * ((k0_perp_0[:, None, None] * r_perp_0[None, :, None]\
                                                + k0_perp_1[:, None, None] * r_perp_1[None, None, :]) * k_unit * dx))
                    # Project and store in a 2D array giving for each k0_0, k0_1 combination the projected wavefunction
                    psi_projection_unfolded = np.tensordot(factor_perp, psi_rot, axes=([1, 2], [axis_id[1], axis_id[2]]))   
                    # determine Fourier transform along each line in the bundle
                    # phase factor array for the separate sections on a line in the bundle
                    kappa = bundle['kappa']
                    factor_par = np.exp(-1j*(k_unit * kappa[:, :, None] * r_par[None, None, :]) * dx) 
                    phi_projection_unfolded = np.tensordot(factor_par, psi_projection_unfolded, axes=([2], [1]))
                    k_segment['phi_folded'] += np.sum(abs(phi_projection_unfolded)**2, axis=(1, 2))  
    log_string = str(n_111) + ' 111 segments processed'
    log_output(log_string, log_file)
    # Transfer the results from all segments to a single 1D array 
    phi_folded_path = organize_output_phi(k_path)
    
    return phi_folded_path

def write_path(project, k_path_names, kappa_ticks, kappa):
    
    fname = project + '_bse_k_path.pkl'
    with open(fname, "wb") as f:
        pickle.dump((k_path_names, kappa_ticks, kappa), f)
    return

def write_bse_folded(project, state_nr, bse_folded_states):
    
    N_states = np.size(state_nr)
    N_i = state_nr[0]
    N_f = state_nr[N_states - 1]
    fname = project + '_' + 'bse_' + 'States_' + str(int(N_i)) + '_' +str(int(N_f)) + '.pkl'
    with open(fname, "wb") as f:
        pickle.dump((state_nr, bse_folded_states), f)
    return    

def hdf5_output(project, k_path_names, kappa_ticks, kappa, state_nr, bse_folded_states, bse_folded_states_binned, E_array):

   N_states = np.size(state_nr)
   N_kappa = np.size(kappa)
   d_kappa = (kappa[1] - kappa[0])
   N_i = state_nr[0]
   N_f = state_nr[N_states - 1]
   fname = project + '_' + 'bse_' + 'States_' + str(int(N_i)) + '_' +str(int(N_f)) + '.h5'
   kappa_ext = kappa - d_kappa/2
   kappa_final = kappa[N_kappa - 1] + d_kappa/2
   kappa_ext = np.concatenate((kappa_ext, [kappa_final]))
   with h5py.File(fname, 'w') as hdf5:
       hdf5.create_dataset('tick_labels', data=k_path_names)
       hdf5.create_dataset('ticks', data=kappa_ticks)
       hdf5.create_dataset('kappa', data=kappa)
       hdf5.create_dataset('kappa_ext', data=kappa_ext)
       hdf5.create_dataset('state_identifier', data=state_nr)
       hdf5.create_dataset('phi_folded', data=bse_folded_states)
       hdf5.create_dataset('phi_folded_binned', data=bse_folded_states_binned)
       hdf5.create_dataset('energy', data=E_array)

def read_energy(file):

    E_list = np.loadtxt(file)

    return E_list

def E_bin(folded_states, E_list, State_0, dE):

    E_list_state_0 = int(E_list[0, 0])
    N_states = np.size(folded_states, axis=0)
    clip_min, clip_max = State_0 - E_list_state_0, State_0 - E_list_state_0 + N_states
    E_list_states = E_list[int(clip_min) : int(clip_max), :]
    E = E_list_states[:, 1]
    # Determine bin edges
    bin_edges = np.arange(E.min(), E.max() + dE, dE)
    # Assign each E value to a bin number
    bin_numbers = np.digitize(E, bin_edges)
    # Create an empty 2D array to store the sums
    E_binned_folded_states = np.zeros((len(bin_edges) - 1, folded_states.shape[1]))
    # Sum rows of States assigned to each bin
    for i in range(len(bin_edges) - 1):
        bin_mask = bin_numbers == (i + 1)
        E_binned_folded_states[i] = np.sum(folded_states[bin_mask], axis=0)
    #print(E_binned_folded_states)
    return bin_edges, E_binned_folded_states


#%% Data structures

""" The e_k_dict contains parameters linked to one of the three relevant direction (100), (110) and (111). Scale is an
often used scaling for the distance, positive_dir sets the directions that will be oriented along a positive direction of 
a spatial axis (and therefore get a positive parallel distance). 'origin_bun' provides the coordinates of one line in a given bundle, 'kappa_0_bun'
gives the shift in parallel wavenumber when labeling a given segment in BZ1 in a given bundle. All this information is transfered
to each segment directory. 
"""

l_110 = 1/np.sqrt(2)
l_111 = 1/np.sqrt(3)
e_k_dict = [{'scale': 1, 'N_bun': 1,\
             'positive_dir': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),\
             'origin_bun': np.array([[0 ,0, 0], [1/2, 1/2, 1/2]])},\
            {'scale': 1/np.sqrt(2), 'N_bun': 1,\
             'positive_dir': np.array([[l_110, l_110, 0], [l_110, -l_110, 0], [0, l_110, l_110],\
                                       [0, l_110, -l_110], [l_110, 0, l_110], [-l_110, 0, l_110]]),\
             'origin_bun': np.array([[0 ,0, 0], [1/2, 1/2, 1/2]])},\
            {'scale': l_111, 'N_bun': 1, 
             'positive_dir': np.array([[l_111, l_111, l_111], [l_111, l_111, -l_111],\
                                       [l_111, -l_111, l_111], [l_111, -l_111, -l_111]]),\
             'origin_bun': np.array([[0 ,0, 0], [1/2, 1/2, 1/2]])}]
